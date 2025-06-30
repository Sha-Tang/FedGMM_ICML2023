import os
from copy import deepcopy

import torch.nn.functional as F

from utils.torch_utils import *


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
                 tune_locally=False, compression_args=None):
        super().__init__(learners_ensemble, train_iterator, val_iterator, test_iterator, logger, local_steps, save_path,
                         tune_locally)
        self.learners_ensemble.initialize_gmm(iterator=train_iterator)
        
        # 🔄 Initialize compression support
        self.compression_args = compression_args
        if hasattr(self.learners_ensemble, 'enable_compression'):
            self.learners_ensemble.enable_compression(compression_args)
            if compression_args and getattr(compression_args, 'use_dgc', False):
                print(f"🔄 Client compression enabled: Top-{compression_args.topk_ratio:.1%}")
        
        # Track communication rounds for compression decision
        self.communication_round = 0

    def update_sample_weights(self):
        self.samples_weights = self.learners_ensemble.calc_samples_weights(self.val_iterator)

    def update_learners_weights(self):  # calculate pi, mu and Var
        self.learners_ensemble.m_step(self.samples_weights, self.val_iterator)
    """
    " Only update gmm
    """
    def step(self, single_batch_flag=False, n_iter=1, *args, **kwargs):
        self.counter += 1
        self.communication_round += 1

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

        # 🔄 Apply communication compression before upload
        compressed_updates = self._apply_compression(client_updates, "classifier")
        
        self.learners_ensemble.free_gradients()
        # self.clear_models()

        return compressed_updates

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

        # 🔄 Apply communication compression for autoencoder updates
        compressed_ac_update = self._apply_compression(ac_client_update, "autoencoder")
        
        return compressed_ac_update

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

    # ========================================
    # 🔄 Communication Compression Methods
    # ========================================
    
    def _apply_compression(self, client_updates, update_type="classifier"):
        """
        Apply communication compression to client updates before upload
        
        Args:
            client_updates: The parameter updates to be uploaded
            update_type: Type of update ("classifier" or "autoencoder")
            
        Returns:
            Compressed updates or original updates if compression disabled
        """
        # Check if compression is enabled
        if not self.compression_args or not getattr(self.compression_args, 'use_dgc', False):
            return client_updates
        
        # Check if learners_ensemble supports compression
        if not hasattr(self.learners_ensemble, 'fit_epochs_with_compression'):
            return client_updates
        
        # Use fit_epochs_with_compression for classifier updates
        if update_type == "classifier" and hasattr(self.learners_ensemble, 'apply_compression_to_updates'):
            try:
                # Convert to tensor for compression processing
                import torch
                client_updates_tensor = torch.tensor(client_updates) if not isinstance(client_updates, torch.Tensor) else client_updates
                
                # Get compression configuration
                compression_info = self.learners_ensemble.get_compressed_params(self.communication_round)
                
                # Apply compression
                compressed_result = self.learners_ensemble.apply_compression_to_updates(
                    client_updates_tensor, compression_info
                )
                
                # Log compression statistics
                self._log_compression_stats(compressed_result, update_type)
                
                return compressed_result
                
            except Exception as e:
                print(f"⚠️  Compression failed for {update_type}: {e}")
                return client_updates
        
        # For autoencoder updates or when compression not supported, apply simple compression
        elif update_type == "autoencoder":
            return self._apply_simple_compression(client_updates, update_type)
        
        return client_updates
    
    def _apply_simple_compression(self, updates, update_type):
        """
        Apply simple compression for non-ensemble updates (like autoencoder)
        
        Args:
            updates: Parameter updates (numpy array)
            update_type: Type of update
            
        Returns:
            Compressed or original updates
        """
        from utils.compression import should_compress, CommunicationCompressor
        
        # Check if we should compress this round
        if not should_compress(self.communication_round, self.compression_args):
            return updates
        
        try:
            import torch
            
            # Create a simple compressor
            compressor = CommunicationCompressor(
                topk_ratio=self.compression_args.topk_ratio,
                strategy=self.compression_args.topk_strategy
            )
            
            # Convert to tensor and compress
            updates_tensor = torch.tensor(updates) if not isinstance(updates, torch.Tensor) else updates
            compressed_values, indices, shapes = compressor.compress(updates_tensor)
            
            # Create compressed result dictionary
            compressed_result = {
                'type': 'compressed',
                'compressed_values': compressed_values.cpu().numpy(),
                'indices': indices.cpu().numpy(),
                'shapes': shapes.cpu().numpy(),
                'compressed': True,
                'round': self.communication_round,
                'compression_ratio': compressor.get_compression_ratio(),
                'update_type': update_type
            }
            
            # Log compression
            self._log_compression_stats(compressed_result, update_type)
            
            return compressed_result
            
        except Exception as e:
            print(f"⚠️  Simple compression failed for {update_type}: {e}")
            return updates
    
    def _log_compression_stats(self, compressed_result, update_type):
        """
        Log compression statistics to TensorBoard
        
        Args:
            compressed_result: Compression result dictionary
            update_type: Type of update being compressed
        """
        if not isinstance(compressed_result, dict) or not compressed_result.get('compressed', False):
            return
        
        try:
            compression_ratio = compressed_result.get('compression_ratio', 1.0)
            round_num = compressed_result.get('round', self.communication_round)
            
            # Log to TensorBoard
            self.logger.add_scalar(f"Compression/{update_type}_ratio", compression_ratio, round_num)
            self.logger.add_scalar("Compression/ratio", compression_ratio, round_num)
            
            # Log communication savings
            communication_savings = (1.0 - compression_ratio) * 100
            self.logger.add_scalar(f"Compression/{update_type}_savings_pct", communication_savings, round_num)
            self.logger.add_scalar("Compression/savings_pct", communication_savings, round_num)
            
            # Print compression info
            print(f"📊 Round {round_num} [{update_type}]: Compressed to {compression_ratio:.1%} "
                  f"(saved {communication_savings:.1f}%)")
                  
        except Exception as e:
            print(f"⚠️  Logging compression stats failed: {e}")
    
    def get_compression_stats(self):
        """
        Get overall compression statistics
        
        Returns:
            Compression statistics dictionary
        """
        if hasattr(self.learners_ensemble, 'get_compression_stats'):
            return self.learners_ensemble.get_compression_stats()
        else:
            return {'compression_enabled': False}


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
