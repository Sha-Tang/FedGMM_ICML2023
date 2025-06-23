import os
from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np

from utils.torch_utils import *
from utils.dgc_compressor import DGCCompressor


class Client(object):
    r"""Implements one clients

    Attributes
    ----------
    learners_ensemble
    n_learners

    train_iterator

    val_iterator

    test_iterator

    train_loader

    n_train_samples

    n_test_samples

    samples_weights

    local_steps

    logger

    tune_locally:

    Methods
    ----------
    __init__
    step
    write_logs
    update_sample_weights
    update_learners_weights

    """

    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            save_path,
            tune_locally=False
    ):

        self.learners_ensemble = learners_ensemble
        self.n_learners = len(self.learners_ensemble)
        self.tune_locally = tune_locally

        if self.tune_locally:
            self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)
        else:
            self.tuned_learners_ensemble = None

        self.binary_classification_flag = self.learners_ensemble.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.samples_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners

        self.local_steps = local_steps

        self.counter = 0
        self.logger = logger

        os.makedirs(save_path, exist_ok=True)
        self.save_path = os.path.join(save_path, 'params.pt')

        # self.clear_models()

    def get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch

    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        # self.reload_models()

        self.counter += 1
        self.update_sample_weights()  # update q(x)
        self.update_learners_weights()  # update pi, mu and Var

        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights
                )

        # TODO: add flag arguments to use `free_gradients`
        self.learners_ensemble.free_gradients()
        # self.clear_models()

        return client_updates

    def clear_models(self):
        self.learners_ensemble.save_models(self.save_path)
        self.learners_ensemble.free_memory()
        self.learners_ensemble.free_gradients()
        torch.cuda.empty_cache()

    def reload_models(self):
        self.learners_ensemble.load_models(self.save_path)

    def write_logs(self):
        if self.tune_locally:
            self.update_tuned_learners()

        if self.tune_locally:
            train_loss, train_acc = self.tuned_learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.tuned_learners_ensemble.evaluate_iterator(self.test_iterator)
        else:
            train_loss, train_acc = self.learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.learners_ensemble.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc

    def update_sample_weights(self):
        pass

    def update_learners_weights(self):
        pass

    def update_tuned_learners(self):
        if not self.tune_locally:
            return

        for learner_id, learner in enumerate(self.tuned_learners_ensemble):
            copy_model(source=self.learners_ensemble[learner_id].model, target=learner.model)
            learner.fit_epochs(self.train_iterator, self.local_steps, weights=self.samples_weights[learner_id])


class MixtureClient(Client):
    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        self.samples_weights = F.softmax((torch.log(torch.Tensor(self.learners_ensemble.learners_weights)) - all_losses.T), dim=1).T

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)


# class GMixtureClient(Client):
#     def update_sample_weights(self):
#         self.samples_weights = self.learners_ensemble.calc_samples_weights(self.val_iterator)
#
#     def update_learners_weights(self):  # calculate pi, mu and Var
#         self.learners_ensemble.m_step(self.samples_weights, self.val_iterator)


class ACGMixtureClient(Client):
    def __init__(self, learners_ensemble, train_iterator, val_iterator, test_iterator, logger, local_steps, save_path,
                 client_id, tune_locally=False, args=None):
        super(ACGMixtureClient, self).__init__(
            learners_ensemble, train_iterator, val_iterator, test_iterator, logger, local_steps, save_path,
            tune_locally
        )
        
        # Initialize GMM for ACG
        self.learners_ensemble.initialize_gmm(iterator=train_iterator)
        
        self.client_id = client_id
        self.args = args
        
        # DGC策略控制参数
        self.use_dgc = args.use_dgc if args else False
        self.compress_ratio = args.compress_ratio if args else 0.3
        self.warmup_rounds = args.warmup_rounds if args else 3
        self.stop_compress_round = args.stop_compress_round if args else -1
        
        # 通信统计（仅客户端0记录）
        self.total_bytes_uploaded = 0
        self.current_round = 0
        
        # DGC压缩器初始化（基于第4步）
        self.dgc_compressor = None
        if self.use_dgc:
            try:
                from utils.dgc_compressor import DGCCompressor
                self.dgc_compressor = DGCCompressor(
                    compress_ratio=self.compress_ratio,
                    momentum=0.9,
                    clipping_norm=1.0
                )
                if self.client_id == 0:
                    print(f"Client {self.client_id}: DGC Strategy Enabled")
                    print(f"  - Compress Ratio: {self.compress_ratio}")
                    print(f"  - Warmup Rounds: {self.warmup_rounds}")
                    print(f"  - Stop Compress Round: {self.stop_compress_round}")
                    print(f"  - DGC Compressor: Initialized")
            except ImportError:
                print(f"Warning: Cannot import DGCCompressor, disabling DGC for client {self.client_id}")
                self.use_dgc = False

    def _should_use_compression(self, current_round):
        """
        第五步核心功能：策略控制逻辑
        判断当前轮是否应该使用压缩
        
        Args:
            current_round: 当前通信轮次
            
        Returns:
            bool: 是否使用压缩
        """
        use_compression = (
            self.use_dgc and  # DGC功能已启用
            current_round >= self.warmup_rounds and  # 超过预热轮数
            (self.stop_compress_round < 0 or current_round < self.stop_compress_round) and  # 未到停止轮数
            self.dgc_compressor is not None  # 压缩器已初始化
        )
        return use_compression
    
    def _calculate_tensor_bytes(self, tensor_or_dict):
        """计算张量或数据结构的字节数"""
        if isinstance(tensor_or_dict, np.ndarray):
            return tensor_or_dict.nbytes
        elif isinstance(tensor_or_dict, dict):
            if tensor_or_dict['type'] == 'dense':
                # 计算稠密模型的总大小
                if hasattr(tensor_or_dict['updates'], 'nbytes'):
                    return tensor_or_dict['updates'].nbytes
                else:
                    return sum(update.nbytes for update in tensor_or_dict['updates'])
            elif tensor_or_dict['type'] == 'compressed':
                # 压缩模型大小 = 原始大小 × 实际压缩比
                total_compressed_size = 0
                for learner_data in tensor_or_dict['learners_data'].values():
                    original_size = learner_data.get('original_size', 0)
                    compressed_size = learner_data.get('compressed_size', 0)
                    # 直接使用实际保留的参数数量 × 4字节（float32）
                    total_compressed_size += compressed_size * 4
                return total_compressed_size
        else:
            return 0
    
    def _log_communication_stats(self, current_round, actual_bytes, dense_bytes, use_compression):
        """
        第五步核心功能：详细的TensorBoard通信统计
        记录详细的通信统计信息到TensorBoard (仅客户端0)
        
        Args:
            current_round: 当前轮次
            actual_bytes: 实际上传字节数
            dense_bytes: 如果不压缩的字节数
            use_compression: 是否使用了压缩
        """
        # 🚀 仅客户端0打印简单上传信息（中文）
        if self.client_id == 0:
            actual_mb = actual_bytes / (1024 * 1024)
            format_type = "压缩" if use_compression else "稠密"
            
            if use_compression:
                dense_mb = dense_bytes / (1024 * 1024)
                compression_ratio = actual_bytes / dense_bytes if dense_bytes > 0 else 1.0
                savings_pct = (1 - compression_ratio) * 100
                print(f"📤 [客户端 0] 第 {current_round} 轮: 上传 {actual_bytes:,} 字节 ({actual_mb:.2f} MB) - {format_type}")
                print(f"   └─ 原始大小: {dense_bytes:,} 字节 ({dense_mb:.2f} MB), 压缩比: {compression_ratio:.3f}, 节省: {savings_pct:.1f}%")
            else:
                print(f"📤 [客户端 0] 第 {current_round} 轮: 上传 {actual_bytes:,} 字节 ({actual_mb:.2f} MB) - {format_type}")
        
        # 仅客户端0记录TensorBoard统计
        if self.client_id != 0:
            return
        
        # 更新累计通信量
        self.total_bytes_uploaded += actual_bytes
        
        # 计算压缩比
        if use_compression and dense_bytes > 0:
            reduction_ratio = actual_bytes / dense_bytes
        else:
            reduction_ratio = 1.0
        
        # 判断当前状态
        is_warmup = current_round < self.warmup_rounds
        is_early_stop = (self.stop_compress_round > 0 and current_round >= self.stop_compress_round)
        
        # 第五步要求的TensorBoard日志记录
        if self.logger:
            # 必须记录的指标
            self.logger.add_scalar('Communication/Round_Bytes', actual_bytes, current_round)
            self.logger.add_scalar('Communication/Total_Bytes', self.total_bytes_uploaded, current_round)
            self.logger.add_scalar('Communication/Reduction_Ratio', reduction_ratio, current_round)
            self.logger.add_scalar('Communication/Compress_Used', int(use_compression), current_round)
            self.logger.add_scalar('Communication/Warmup_Phase', int(is_warmup), current_round)
            self.logger.add_scalar('Communication/Early_Stop_Phase', int(is_early_stop), current_round)
            
            # 额外有用的统计信息
            self.logger.add_scalar('Communication/Dense_Bytes', dense_bytes, current_round)
            if use_compression:
                compression_savings = dense_bytes - actual_bytes
                self.logger.add_scalar('Communication/Savings_Bytes', compression_savings, current_round)
                self.logger.add_scalar('Communication/Savings_Percentage', (compression_savings / dense_bytes) * 100, current_round)
                
            # 记录策略状态
            self.logger.add_scalar('Communication/Current_Round', current_round, current_round)
            
        # 控制台输出（前几轮或每10轮）- 显示理论压缩效果
        if current_round <= 5 or current_round % 10 == 0:
            phase_name = "预热" if is_warmup else ("早停" if is_early_stop else "活跃")
            if use_compression:
                theoretical_ratio = self.compress_ratio  # 理论压缩比
                actual_ratio = actual_bytes / dense_bytes if dense_bytes > 0 else 1.0
                print(f"[客户端 0] 第 {current_round} 轮 ({phase_name}): "
                      f"模型大小={actual_bytes:,}字节 (压缩), "
                      f"理论压缩比={theoretical_ratio:.1%}, "
                      f"实际压缩比={actual_ratio:.3f}, "
                      f"节省={(1-actual_ratio)*100:.1f}%")
            else:
                print(f"[客户端 0] 第 {current_round} 轮 ({phase_name}): "
                      f"模型大小={actual_bytes:,}字节 (稠密), "
                      f"累计={self.total_bytes_uploaded:,}字节")

    def _compress_updates_for_upload(self, client_updates, current_round):
        """
        基于第4步的DGC压缩器，添加第五步的策略控制
        根据策略决定是否压缩更新并返回上传格式
        
        Args:
            client_updates: 客户端模型更新 (numpy array)
            current_round: 当前通信轮次
            
        Returns:
            dict: 上传数据格式 (dense 或 compressed)
        """
        # 第五步策略控制逻辑
        use_compression = self._should_use_compression(current_round)
        
        # 计算dense格式的字节数（用于统计）
        dense_bytes = self._calculate_tensor_bytes(client_updates)
        
        if use_compression:
            # 使用第4步实现的实际DGC压缩器
            compressed_data = {}
            for learner_id in range(len(client_updates)):
                learner_update = client_updates[learner_id]
                
                # 转换为torch tensor进行压缩
                grad_tensor = torch.from_numpy(learner_update).float()
                indices, values, shape = self.dgc_compressor.step(grad_tensor, learner_id)
                
                # 转换回numpy并保存
                compressed_data[learner_id] = {
                    'indices': indices.cpu().numpy(),
                    'values': values.cpu().numpy(), 
                    'shape': shape,
                    'original_size': int(learner_update.size),
                    'compressed_size': int(len(values)),
                    'compression_ratio': float(len(values) / learner_update.size)
                }
            
            upload_data = {
                'type': 'compressed',
                'learners_data': compressed_data,
                'compressed': True,
                'client_id': self.client_id
            }
            
            actual_bytes = self._calculate_tensor_bytes(upload_data)
            
        else:
            # 不使用压缩，上传dense格式
            upload_data = {
                'type': 'dense',
                'updates': client_updates,
                'compressed': False,
                'client_id': self.client_id
            }
            
            actual_bytes = dense_bytes
        
        # 第五步要求的通信统计记录
        self._log_communication_stats(current_round, actual_bytes, dense_bytes, use_compression)
        
        return upload_data

    def step(self, current_round=None, single_batch_flag=False, *args, **kwargs):
        """
        客户端训练步骤，整合第4步的DGC功能和第5步的策略控制
        
        Args:
            current_round: 当前通信轮次
            single_batch_flag: 是否只训练一个batch
            
        Returns:
            dict: 上传数据 (dense或compressed格式)
        """
        if current_round is not None:
            self.current_round = current_round
        
        # 执行本地训练
        sum_samples_weights = torch.zeros(self.n_learners, dtype=torch.float)
        for learner in self.learners_ensemble:
            sum_samples_weights += learner.get_sample_weights()
        sum_samples_weights = sum_samples_weights / self.n_learners

        if single_batch_flag:
            client_updates = self.learners_ensemble.fit_batch(weights=sum_samples_weights)
        else:
            client_updates = self.learners_ensemble.fit_epochs(
                iterator=self.train_iterator,
                n_epochs=self.local_steps,
                weights=sum_samples_weights
            )

        # 应用第五步的策略控制和通信统计
        return self._compress_updates_for_upload(client_updates, self.current_round)

    def update_sample_weights(self):
        self.samples_weights = self.learners_ensemble.calc_samples_weights(self.val_iterator)

    def update_learners_weights(self):  # calculate pi, mu and Var
        self.learners_ensemble.m_step(self.samples_weights, self.val_iterator)
    """
    " Only update gmm
    """
    def step(self, single_batch_flag=False, n_iter=1, current_round=None, *args, **kwargs):
        self.counter += 1
        
        # Update current round for DGC warmup control
        if current_round is not None:
            self.current_round = current_round

        # self.learners_ensemble.initialize_gmm(iterator=self.train_iterator)
        """
        " EM step
        """
        for _ in range(n_iter):
            self.update_sample_weights()  # update q(x)
            self.update_learners_weights()  # update pi, mu and Var

        sum_samples_weights = self.samples_weights.sum(dim=1)
        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=sum_samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=sum_samples_weights
                )

        # DGC compression for upload
        compressed_updates = self._compress_updates_for_upload(client_updates, self.current_round)

        self.learners_ensemble.free_gradients()
        # self.clear_models()

        return compressed_updates
    # 被注释掉的源代码
    '''
    def _compress_updates_for_upload(self, client_updates):
        """
        Compress client updates using DGC if enabled and not in warmup period.
        
        Args:
            client_updates: numpy array of shape (n_learners, model_dim)
            
        Returns:
            dict: Either dense updates or compressed sparse representation
        """
        # Check if compression should be used
        use_compression = (
            self.use_dgc and 
            self.current_round >= self.warmup_rounds and 
            self.dgc_compressor is not None
        )
        
        if not use_compression:
            # Dense upload during warmup or when DGC is disabled
            total_bytes = self._calculate_dense_comm_size(client_updates)
            self._log_communication_stats(total_bytes, use_compression=False)
            
            return {
                'type': 'dense',
                'updates': client_updates,
                'compressed': False
            }
        
        # Compress each learner's updates
        compressed_data = {}
        total_compressed_bytes = 0
        total_dense_bytes = 0
        
        for learner_id in range(len(client_updates)):
            grad_tensor = torch.from_numpy(client_updates[learner_id]).float()
            
            # Calculate original size
            dense_bytes = grad_tensor.numel() * 4  # 4 bytes per float32
            total_dense_bytes += dense_bytes
            
            # Compress using DGC
            indices, values, shape = self.dgc_compressor.step(grad_tensor, learner_id)
            
            # Calculate compressed size (indices + values)
            compressed_bytes = (len(indices) + len(values)) * 4
            total_compressed_bytes += compressed_bytes
            
            compressed_data[learner_id] = {
                'indices': indices.cpu().numpy(),
                'values': values.cpu().numpy(), 
                'shape': shape,
                'dense_bytes': dense_bytes,
                'compressed_bytes': compressed_bytes
            }
        
        # Log communication statistics
        self._log_communication_stats(total_compressed_bytes, use_compression=True, 
                                    original_bytes=total_dense_bytes)
        
        return {
            'type': 'compressed',
            'learners_data': compressed_data,
            'compressed': True,
            'total_compressed_bytes': total_compressed_bytes,
            'total_dense_bytes': total_dense_bytes
        }
    
    def _calculate_dense_comm_size(self, client_updates):
        """Calculate total communication size for dense updates."""
        total_bytes = 0
        for learner_id in range(len(client_updates)):
            total_bytes += client_updates[learner_id].size * 4  # 4 bytes per float32
        return total_bytes
    
    def _log_communication_stats(self, comm_bytes, use_compression=False, original_bytes=None):
        """Log communication statistics to TensorBoard (only for client 0)."""
        if self.client_id != 0:  # Only log for the first client
            return
            
        # Update statistics
        self.comm_stats['round_comm_bytes'].append(comm_bytes)
        
        if use_compression and original_bytes:
            self.comm_stats['total_compressed_bytes'] += comm_bytes
            self.comm_stats['total_dense_bytes'] += original_bytes
            compression_ratio = comm_bytes / original_bytes
            self.comm_stats['compression_ratios'].append(compression_ratio)
        else:
            self.comm_stats['total_dense_bytes'] += comm_bytes
        
        # Log to TensorBoard
        self.logger.add_scalar('Communication/Round_Bytes', comm_bytes, self.current_round)
        self.logger.add_scalar('Communication/Avg_Bytes_Per_Learner', 
                             comm_bytes / self.n_learners, self.current_round)
        
        if use_compression:
            self.logger.add_scalar('Communication/Compression_Enabled', 1.0, self.current_round)
            if original_bytes:
                reduction_ratio = 1.0 - (comm_bytes / original_bytes)
                self.logger.add_scalar('Communication/Reduction_Ratio', 
                                     reduction_ratio, self.current_round)
                self.logger.add_scalar('Communication/Original_Bytes', 
                                     original_bytes, self.current_round)
        else:
            self.logger.add_scalar('Communication/Compression_Enabled', 0.0, self.current_round)
        
        # Log cumulative statistics
        if len(self.comm_stats['round_comm_bytes']) > 0:
            total_comm = sum(self.comm_stats['round_comm_bytes'])
            avg_comm_per_round = total_comm / len(self.comm_stats['round_comm_bytes'])
            self.logger.add_scalar('Communication/Cumulative_Total_Bytes', total_comm, self.current_round)
            self.logger.add_scalar('Communication/Avg_Bytes_Per_Round', avg_comm_per_round, self.current_round)
        
        if self.use_dgc and len(self.comm_stats['compression_ratios']) > 0:
            avg_compression_ratio = sum(self.comm_stats['compression_ratios']) / len(self.comm_stats['compression_ratios'])
            self.logger.add_scalar('Communication/Avg_Compression_Ratio', avg_compression_ratio, self.current_round)
'''
    def unseen_step(self, single_batch_flag=False, n_iter=1, *args, **kwargs):
        self.counter += 1

        """
        " EM step
        """
        for _ in range(n_iter):
            self.update_sample_weights()  # update q(x)

    def gmm_step(self, single_batch_flag=False, n_iter=1, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        # self.reload_models()
        self.counter += 1

        # self.learners_ensemble.initialize_gmm(iterator=self.train_iterator)
        """
        " EM step
        """
        for _ in range(n_iter):
            self.update_sample_weights()  # update q(x)
            self.update_learners_weights()  # update pi, mu and Var

        # sum_samples_weights = self.samples_weights.sum(dim=1)
        # if single_batch_flag:
        #     batch = self.get_next_batch()
        #     client_updates = \
        #         self.learners_ensemble.fit_batch(
        #             batch=batch,
        #             weights=sum_samples_weights
        #         )
        # else:
        #     client_updates = \
        #         self.learners_ensemble.fit_epochs(
        #             iterator=self.train_iterator,
        #             n_epochs=self.local_steps,
        #             weights=sum_samples_weights
        #         )

        # self.learners_ensemble.free_gradients()
        # self.clear_models()

        # return client_updates
        return

    def ac_step(self):
        self.learners_ensemble.freeze_classifier()

        ac_client_update = \
            self.learners_ensemble.fit_ac_epochs(
                iterator=self.train_iterator,
                n_epochs=self.local_steps
            )

        self.learners_ensemble.unfreeze_classifier()

        return ac_client_update

    def write_logs(self):
        if self.tune_locally:
            self.update_tuned_learners()

        if self.tune_locally:
            train_loss, train_acc = self.tuned_learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.tuned_learners_ensemble.evaluate_iterator(self.test_iterator)
        else:
            train_loss, train_acc = self.learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.learners_ensemble.evaluate_iterator(self.test_iterator)
            # train_recon, train_nll = self.learners_ensemble.evaluate_ac_iterator(self.val_iterator)
            # test_recon, test_nll = self.learners_ensemble.evaluate_ac_iterator(self.test_iterator)

        train_recon = 0
        train_nll = 0
        test_recon = 0
        test_nll = 0
        #not used
        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        self.logger.add_scalar("Train/Recon_Loss", train_recon, self.counter)
        self.logger.add_scalar("Train/NLL", train_nll, self.counter)
        self.logger.add_scalar("Test/Recon_Loss", test_recon, self.counter)
        self.logger.add_scalar("Test/NLL", test_nll, self.counter)

        return train_loss, train_acc, test_loss, test_acc, train_recon, train_nll, test_recon, test_nll


class AgnosticFLClient(Client):
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False
    ):
        super(AgnosticFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."

    def step(self, *args, **kwargs):
        self.counter += 1

        batch = self.get_next_batch()
        losses = self.learners_ensemble.compute_gradients_and_loss(batch)

        return losses


class FFLClient(Client):
    r"""
    Implements client for q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """

    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            q=1,
            tune_locally=False
    ):
        super(FFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."
        self.q = q

    def step(self, lr, *args, **kwargs):
        hs = 0
        for learner in self.learners_ensemble:
            initial_state_dict = self.learners_ensemble[0].model.state_dict()
            learner.fit_epochs(iterator=self.train_iterator, n_epochs=self.local_steps)

            client_loss, _ = learner.evaluate_iterator(self.train_iterator)
            client_loss = torch.tensor(client_loss)
            client_loss += 1e-10

            # assign the difference to param.grad for each param in learner.parameters()
            differentiate_learner(
                target=learner,
                reference_state_dict=initial_state_dict,
                coeff=torch.pow(client_loss, self.q) / lr
            )

            hs = self.q * torch.pow(client_loss, self.q - 1) * torch.pow(torch.linalg.norm(learner.get_grad_tensor()),
                                                                         2)
            hs /= torch.pow(torch.pow(client_loss, self.q), 2)
            hs += torch.pow(client_loss, self.q) / lr

        return hs / len(self.learners_ensemble)
