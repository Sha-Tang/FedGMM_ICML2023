import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .gmm import GaussianMixture


class LearnersEnsemble(object):
    """
    Iterable Ensemble of Learners.

    Attributes
    ----------
    learners
    learners_weights
    model_dim
    is_binary_classification
    device
    metric

    Methods
    ----------
    __init__
    __iter__
    __len__
    compute_gradients_and_loss
    optimizer_step
    fit_epochs
    evaluate
    gather_losses
    free_memory
    free_gradients

    """

    def __init__(self, learners, learners_weights):
        self.learners = learners
        self.learners_weights = learners_weights

        self.model_dim = self.learners[0].model_dim
        self.is_binary_classification = self.learners[0].is_binary_classification
        self.device = self.learners[0].device
        self.metric = self.learners[0].metric

    def optimizer_step(self):
        """
        perform one optimizer step, requires the gradients to be already computed
        """
        for learner in self.learners:
            learner.optimizer_step()

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        losses = []
        for learner_id, learner in enumerate(self.learners):
            loss = learner.compute_gradients_and_loss(batch, weights=weights)
            losses.append(loss)

        return losses

    def fit_batch(self, batch, weights):
        """
        updates learners using  one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
            and the updated parameters for each learner in the ensemble.

        """
        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):
            old_params = learner.get_param_tensor()
            if weights is not None:
                learner.fit_batch(batch=batch, weights=weights[learner_id])
            else:
                learner.fit_batch(batch=batch, weights=None)

            params = learner.get_param_tensor()

            client_updates[learner_id] = (params - old_params)

        return client_updates.cpu().numpy()

    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        perform multiple training epochs, updating each learner in the ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of epochs
        :type n_epochs: int
        :param weights: tensor of shape (n_learners, len(iterator)), holding the weight of each sample in iterator
                        for each learner ins ensemble_learners
        :type weights: torch.tensor or None
        :return:
            client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
            and the updated parameters for each learner in the ensemble.

        """
        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):
            old_params = learner.get_param_tensor()
            if weights is not None:
                learner.fit_epochs(iterator, n_epochs, weights=weights[learner_id])
            else:
                learner.fit_epochs(iterator, n_epochs, weights=None)
            params = learner.get_param_tensor()

            client_updates[learner_id] = (params - old_params)

        return client_updates.cpu().numpy()

    def evaluate_iterator(self, iterator):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """
        if self.is_binary_classification:
            criterion = nn.BCELoss(reduction="none")
        else:
            criterion = nn.NLLLoss(reduction="none")

        for learner in self.learners:
            learner.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)
                n_samples += y.size(0)

                y_pred = 0.
                for learner_id, learner in enumerate(self.learners):
                    if self.is_binary_classification:
                        y_pred += self.learners_weights[learner_id] * torch.sigmoid(learner.model(x))
                    else:
                        y_pred += self.learners_weights[learner_id] * F.softmax(learner.model(x), dim=1)

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)
                    global_loss += criterion(y_pred, y).sum().item()
                    y_pred = torch.logit(y_pred, eps=1e-10)
                else:
                    global_loss += criterion(torch.log(y_pred), y).sum().item()

                global_metric += self.metric(y_pred, y).item()

            return global_loss / n_samples, global_metric / n_samples

    def gather_losses(self, iterator):
        """
        gathers losses for all sample in iterator for each learner in ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor (n_learners, n_samples) with losses of all elements of the iterator.dataset

        """
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(len(self.learners), n_samples)
        for learner_id, learner in enumerate(self.learners):
            all_losses[learner_id] = learner.gather_losses(iterator)

        return all_losses

    def save_models(self, path):
        para_dict = []
        for learner in self.learners:
            para_dict.append(learner.model.state_dict())
        torch.save(para_dict, path)

    def load_models(self, path):
        para_dict = torch.load(path)
        for learner_id, learner in self.learners:
            learner.model.load_state_dict(para_dict[learner_id])

    def free_memory(self):
        """
        free_memory: free the memory allocated by the model weights

        """
        for learner in self.learners:
            learner.free_memory()

    def free_gradients(self):
        """
        free memory allocated by gradients

        """
        for learner in self.learners:
            learner.free_gradients()

    def __iter__(self):
        return LearnersEnsembleIterator(self)

    def __len__(self):
        return len(self.learners)

    def __getitem__(self, idx):
        return self.learners[idx]


class LanguageModelingLearnersEnsemble(LearnersEnsemble):
    def evaluate_iterator(self, iterator):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """
        criterion = nn.NLLLoss(reduction="none")

        for learner in self.learners:
            learner.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device)
                y = y.to(self.device)
                n_samples += y.size(0)
                chunk_len = y.size(1)

                y_pred = 0.
                for learner_id, learner in enumerate(self.learners):
                    y_pred += self.learners_weights[learner_id] * F.softmax(learner.model(x), dim=1)

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                global_loss += criterion(torch.log(y_pred), y).sum().item() / chunk_len
                global_metric += self.metric(y_pred, y).item() / chunk_len

            return global_loss / n_samples, global_metric / n_samples


class LearnersEnsembleIterator(object):
    """
    LearnersEnsemble iterator class

    Attributes
    ----------
    _learners_ensemble
    _index

    Methods
    ----------
    __init__
    __next__

    """

    def __init__(self, learners_ensemble):
        self._learners_ensemble = learners_ensemble.learners
        self._index = 0

    def __next__(self):
        while self._index < len(self._learners_ensemble):
            result = self._learners_ensemble[self._index]
            self._index += 1

            return result

        raise StopIteration


# class GLearnersEnsemble(object):
#
#     def __init__(self, learners, learners_weights, embedding_dim):
#         self.learners = learners
#         self.learners_weights = learners_weights
#
#         self.model_dim = self.learners[0].model_dim
#         self.is_binary_classification = self.learners[0].is_binary_classification
#         self.device = self.learners[0].device
#         self.metric = self.learners[0].metric
#
#         self.gmm = GaussianMixture(n_components=len(learners), n_features=embedding_dim, device=self.device)
#         self.learners_weights = self.learners_weights.to(self.device)
#
#     def optimizer_step(self):
#         """
#         perform one optimizer step, requires the gradients to be already computed
#         """
#         for learner in self.learners:
#             learner.optimizer_step()
#
#     def compute_gradients_and_loss(self, batch, weights=None):
#         """
#         compute the gradients and loss over one batch.
#
#         :param batch: tuple of (x, y, indices)
#         :param weights: tensor with the learners_weights of each sample or None
#         :type weights: torch.tensor or None
#         :return:
#             loss
#
#         """
#         losses = []
#         for learner_id, learner in enumerate(self.learners):
#             loss = learner.compute_gradients_and_loss(batch, weights=weights)
#             losses.append(loss)
#
#         return losses
#
#     def fit_batch(self, batch, weights):
#         """
#         updates learners using  one batch.
#
#         :param batch: tuple of (x, y, indices)
#         :param weights: tensor with the learners_weights of each sample or None
#         :type weights: torch.tensor or None
#         :return:
#             client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
#             and the updated parameters for each learner in the ensemble.
#
#         """
#         client_updates = torch.zeros(len(self.learners), self.model_dim)
#
#         for learner_id, learner in enumerate(self.learners):
#             old_params = learner.get_param_tensor()
#             if weights is not None:
#                 learner.fit_batch(batch=batch, weights=weights[learner_id])
#             else:
#                 learner.fit_batch(batch=batch, weights=None)
#
#             params = learner.get_param_tensor()
#
#             client_updates[learner_id] = (params - old_params)
#
#         return client_updates.cpu().numpy()
#
#     def fit_epochs(self, iterator, n_epochs, weights=None):
#         """
#         perform multiple training epochs, updating each learner in the ensemble
#
#         :param iterator:
#         :type iterator: torch.utils.data.DataLoader
#         :param n_epochs: number of epochs
#         :type n_epochs: int
#         :param weights: tensor of shape (n_learners, len(iterator)), holding the weight of each sample in iterator
#                         for each learner ins ensemble_learners
#         :type weights: torch.tensor or None
#         :return:
#             client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
#             and the updated parameters for each learner in the ensemble.
#
#         """
#         client_updates = torch.zeros(len(self.learners), self.model_dim)
#
#         for learner_id, learner in enumerate(self.learners):
#             old_params = learner.get_param_tensor()
#             if weights is not None:
#                 learner.fit_epochs(iterator, n_epochs, weights=weights[learner_id])
#             else:
#                 learner.fit_epochs(iterator, n_epochs, weights=None)
#             params = learner.get_param_tensor()
#
#             client_updates[learner_id] = (params - old_params)
#
#         return client_updates.cpu().numpy()
#
#     def evaluate_iterator(self, iterator):
#         """
#         Evaluate a ensemble of learners on iterator.
#
#         :param iterator: yields x, y, indices
#         :type iterator: torch.utils.data.DataLoader
#         :return: global_loss, global_acc
#
#         """
#         if self.is_binary_classification:
#             criterion = nn.BCELoss(reduction="none")
#         else:
#             criterion = nn.NLLLoss(reduction="none")
#
#         for learner in self.learners:
#             learner.model.eval()
#
#         global_loss = 0.
#         global_metric = 0.
#         n_samples = 0
#
#         with torch.no_grad():
#             for (x, y, _) in iterator:
#                 x = x.to(self.device).type(torch.float32)
#                 y = y.to(self.device)
#                 n_samples += y.size(0)
#
#                 p_k_x = self.gmm.predict(x, probs=True) #normalized p(x,k) \prop p(k) * p(x|k) = pi_k * N_k(x)
#
#                 y_pred = 0.
#                 p_x_pred = 0.
#                 for learner_id, learner in enumerate(self.learners):
#                     if self.is_binary_classification:
#                         y_pred += p_k_x[:, learner_id] * torch.sigmoid(
#                             learner.model(x))
#                         p_x_pred += p_k_x[:, learner_id]
#                     else:
#                         y_pred += p_k_x[:, learner_id][:, None] * F.softmax(
#                             learner.model(x), dim=1)
#                         p_x_pred += p_k_x[:, learner_id]
#
#                 y_pred = y_pred / p_x_pred[:, None]
#
#                 y_pred = torch.clamp(y_pred, min=0., max=1.)
#
#                 if self.is_binary_classification:
#                     y = y.type(torch.float32).unsqueeze(1)
#                     global_loss += criterion(y_pred, y).sum().item()
#                     y_pred = torch.logit(y_pred, eps=1e-10)
#                 else:
#                     global_loss += criterion(torch.log(y_pred), y).sum().item()
#
#                 global_metric += self.metric(y_pred, y).item()
#
#             return global_loss / n_samples, global_metric / n_samples
#
#     def calc_log_prob_y_x_batch(self, x, y):
#         """
#         gathers losses for all sample in iterator for each learner in ensemble
#
#         :param iterator:
#         :type iterator: torch.utils.data.DataLoader
#         :return
#             tensor (n_samples, n_learners) with losses of all elements of the iterator.dataset
#
#         """
#         n_samples = x.size()[0]
#         all_losses = torch.zeros(len(self.learners), n_samples).to(self.device)
#         for learner_id, learner in enumerate(self.learners):
#             all_losses[learner_id] = learner.calc_log_prob_batch(x, y)
#
#         return -all_losses
#
#     def gather_losses(self, iterator):
#         """
#         gathers losses for all sample in iterator for each learner in ensemble
#
#         :param iterator:
#         :type iterator: torch.utils.data.DataLoader
#         :return
#             tensor (n_learners, n_samples) with losses of all elements of the iterator.dataset
#
#         """
#         n_samples = len(iterator.dataset)
#         all_losses = torch.zeros(len(self.learners), n_samples).to(self.device)
#         for learner_id, learner in enumerate(self.learners):
#             all_losses[learner_id] = learner.gather_losses(iterator)
#
#         return all_losses
#
#     def free_memory(self):
#         """
#         free_memory: free the memory allocated by the model weights
#
#         """
#         for learner in self.learners:
#             learner.free_memory()
#
#     def free_gradients(self):
#         """
#         free memory allocated by gradients
#
#         """
#         for learner in self.learners:
#             learner.free_gradients()
#
#     def calc_samples_weights(self, iterator):
#         assert torch.equal(self.learners_weights.to('cpu'), self.gmm.pi[0, :, 0].to('cpu')), \
#             "Discrepancy between learner weights!"
#         all_losses = self.gather_losses(iterator)
#
#         with torch.no_grad():
#             x = iterator.dataset.data
#             x = x.to(self.device).type(torch.float32)
#             all_log_prob = self.gmm.calc_log_prob(x)
#
#             sample_weights = F.softmax((torch.log(self.learners_weights) + all_log_prob - all_losses.T), dim=1).T
#             # sample_weights = F.softmax((torch.log(self.learners_weights) + all_log_prob), dim=1).T
#
#         return sample_weights
#
#     def m_step(self, sample_weights, iterator):
#         x = iterator.dataset.data
#         x = x.to(self.device).type(torch.float32)
#
#         sample_weights = sample_weights.unsqueeze(2).transpose(0, 1).to(self.device)
#
#         pi, mu, var = self.gmm.m_step_with_response(x, sample_weights)
#
#         self.learners_weights = convert_pi(pi)
#
#     def freeze_classifier(self):
#         for learner in self.learners:
#             learner.freeze()
#
#     def unfreeze_classifier(self):
#         for learner in self.learners:
#             learner.unfreeze()
#
#     def __iter__(self):
#         return LearnersEnsembleIterator(self)
#
#     def __len__(self):
#         return len(self.learners)
#
#     def __getitem__(self, idx):
#         return self.learners[idx]


class ACGLearnersEnsemble(object):

    def __init__(self, learners, embedding_dim, autoencoder, n_gmm):
        self.learners = learners

        self.n_learners = len(learners)
        self.n_gmm = n_gmm
        self.learners_weights = torch.ones(self.n_gmm, self.n_learners) / (self.n_learners * self.n_gmm)

        self.model_dim = self.learners[0].model_dim
        self.is_binary_classification = self.learners[0].is_binary_classification
        self.device = self.learners[0].device
        self.metric = self.learners[0].metric

        self.gmm = GaussianMixture(n_components=self.n_gmm, n_features=embedding_dim, device=self.device)
        self.learners_weights = self.learners_weights.to(self.device)

        self.autoencoder = autoencoder

        self.reconstruction_weight = 10.0
        self.nll_weight = 1.0

        self.first_step = True

    def optimizer_step(self):
        """
        perform one optimizer step, requires the gradients to be already computed
        """
        for learner in self.learners:
            learner.optimizer_step()

    def compute_gradients_and_loss(self, batch, weights=None):
        """
        compute the gradients and loss over one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            loss

        """
        losses = []
        for learner_id, learner in enumerate(self.learners):
            loss = learner.compute_gradients_and_loss(batch, weights=weights)
            losses.append(loss)

        return losses

    def fit_batch(self, batch, weights=None):
        """
        updates learners using  one batch.

        :param batch: tuple of (x, y, indices)
        :param weights: tensor with the learners_weights of each sample or None
        :type weights: torch.tensor or None
        :return:
            client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
            and the updated parameters for each learner in the ensemble.

        """
        client_updates = torch.zeros(len(self.learners), self.model_dim)

        for learner_id, learner in enumerate(self.learners):
            old_params = learner.get_param_tensor()
            if weights is not None:
                wT = weights[:, learner_id].view([-1])
                learner.fit_batch(batch=batch, weights=wT)
            else:
                learner.fit_batch(batch=batch, weights=None)

            params = learner.get_param_tensor()

            client_updates[learner_id] = (params - old_params)

        return client_updates.cpu().numpy()

    def fit_epochs(self, iterator, n_epochs, weights=None):
        """
        perform multiple training epochs, updating each learner in the ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :param n_epochs: number of epochs
        :type n_epochs: int
        :param weights: tensor of shape (n_learners, len(iterator)), holding the weight of each sample in iterator
                        for each learner ins ensemble_learners
        :type weights: torch.tensor or None
        :return:
            client_updates (np.array, shape=(n_learners, model_dim)): the difference between the old parameter
            and the updated parameters for each learner in the ensemble.

        """
        client_updates = torch.zeros(len(self.learners), self.model_dim)


        for learner_id, learner in enumerate(self.learners):
            old_params = learner.get_param_tensor()
            if weights is not None:
                wT = weights[:, learner_id].view([-1])
                learner.fit_epochs(iterator, n_epochs, weights=wT)
            else:
                learner.fit_epochs(iterator, n_epochs, weights=None)
            params = learner.get_param_tensor()

            client_updates[learner_id] = (params - old_params)

        return client_updates.cpu().numpy()

    def evaluate_iterator(self, iterator):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """
        if self.is_binary_classification:
            criterion = nn.BCELoss(reduction="none")
        else:
            criterion = nn.NLLLoss(reduction="none")

        for learner in self.learners:
            learner.model.eval()

        global_loss = 0.
        global_metric = 0.
        n_samples = 0

        with torch.no_grad():
            for (x, y, _) in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)
                n_samples += y.size(0)

                p_k_x = self.predict_gmm(x).sum(dim=1)  # n * m2
                assert not torch.isnan(p_k_x).any()

                y_pred = 0.
                p_x_pred = 0.
                for learner_id, learner in enumerate(self.learners):
                    if self.is_binary_classification:
                        y_pred += p_k_x[:, learner_id].unsqueeze(1) * torch.sigmoid(
                            learner.model(x))
                        p_x_pred += p_k_x[:, learner_id]
                    else:
                        y_pred += p_k_x[:, learner_id].unsqueeze(1) * F.softmax(
                            learner.model(x), dim=1)
                        p_x_pred += p_k_x[:, learner_id]

                # assert (p_x_pred == 1.0).all()
                # y_pred = y_pred / p_x_pred[:, None]

                y_pred = torch.clamp(y_pred, min=0., max=1.)

                if self.is_binary_classification:
                    y = y.type(torch.float32).unsqueeze(1)
                    global_loss += criterion(y_pred, y).sum().item()
                    y_pred = torch.logit(y_pred, eps=1e-10)
                else:
                    assert not torch.isnan(criterion(torch.log(y_pred), y).sum())
                    global_loss += criterion(torch.log(y_pred), y).sum().item()

                global_metric += self.metric(y_pred, y).item()

            return global_loss / n_samples, global_metric / n_samples

    def evaluate_batch(self, batch):
        """
        Evaluate a ensemble of learners on iterator.

        :param iterator: yields x, y, indices
        :type iterator: torch.utils.data.DataLoader
        :return: global_loss, global_acc

        """
        if self.is_binary_classification:
            criterion = nn.BCELoss(reduction="none")
        else:
            criterion = nn.NLLLoss(reduction="none")

        for learner in self.learners:
            learner.model.eval()

        with torch.no_grad():
            x, y = batch
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            p_k_x = self.predict_gmm(x).sum(dim=1)  # n * m2
            assert not torch.isnan(p_k_x).any()

            y_pred = 0.
            p_x_pred = 0.
            for learner_id, learner in enumerate(self.learners):
                if self.is_binary_classification:
                    y_pred += p_k_x[:, learner_id].unsqueeze(1) * torch.sigmoid(
                        learner.model(x))
                    p_x_pred += p_k_x[:, learner_id]
                else:
                    y_pred += p_k_x[:, learner_id][:, None] * F.softmax(
                        learner.model(x), dim=1)
                    p_x_pred += p_k_x[:, learner_id]

            # assert (p_x_pred != 0).all()
            # y_pred = y_pred / p_x_pred[:, None]

            y_pred = torch.clamp(y_pred, min=0., max=1.)

            if self.is_binary_classification:
                y = y.type(torch.float32).unsqueeze(1)
                losses = criterion(y_pred, y)
                y_pred = torch.logit(y_pred, eps=1e-10)
            else:
                assert not torch.isnan(criterion(torch.log(y_pred), y).sum())
                losses = criterion(torch.log(y_pred), y)

        return -losses

    def predict_gmm(self, x):  # x must be a batch, return th probability of x belong to component k
        self.autoencoder.model.eval()
        with torch.no_grad():
            z = self.autoencoder.model.encode(x)
            log_prob_gmm = self.gmm.calc_log_prob(z).unsqueeze(2)  # n * m1 * 1
            weighted_log_prob = log_prob_gmm + self.learners_weights.unsqueeze(0)  # n * m1 * m2

            prob = torch.softmax(weighted_log_prob.view([-1, self.n_gmm * self.n_learners]), dim=1)
        return prob.view([-1, self.n_gmm, self.n_learners])  # n * m1 * m2

    def calc_log_prob_gmm(self, x):  # x must be a batch
        self.autoencoder.model.eval()
        with torch.no_grad():
            z = self.autoencoder.model.encode(x)

        return self.gmm.calc_log_prob(z)

    def initialize_gmm(self, iterator):
        if self.first_step:
            self.first_step = False

            self.autoencoder.model.eval()
            with torch.no_grad():
                data = []
                for (x, y, _) in iterator:
                    x = x.to(self.device).type(torch.float32)
                    x_rep = self.autoencoder.model.encode(x)
                    data.append(x_rep)

            data = torch.cat(data)

            self.gmm.initialize_gmm(data)

            # self.learners_weights = convert_pi(pi)

            # w = self.calc_samples_weights(iterator)
        return

    def calc_samples_weights(self, iterator):
        # assert torch.equal(self.learners_weights.sum(dim=0).to('cpu'), self.gmm.pi[0, :, 0].to('cpu')), \
        #     "Discrepancy between learner weights!, learners weights: {} pi in gmm: {}".format(
        #         self.learners_weights.sum(dim=0).to('cpu'),
        #         self.gmm.pi[0, :, 0].to('cpu'))
        all_losses = self.gather_losses(iterator).T  # n * m2

        with torch.no_grad():
            n_samples = len(iterator.dataset)
            all_log_prob = torch.zeros(n_samples, self.n_gmm, device=self.device)  # n * m1
            for (x, y, indices) in iterator:
                x = x.to(self.device).type(torch.float32)
                log_prob = self.calc_log_prob_gmm(x)
                all_log_prob[indices, :] = log_prob

            # sample_weights = F.softmax((torch.log(self.learners_weights) + all_log_prob - all_losses.T), dim=1).T
            # sample_weights = F.softmax((torch.log(self.learners_weights) + all_log_prob), dim=1).T
            weighted_log = torch.log(self.learners_weights).unsqueeze(0) + all_log_prob.unsqueeze(2) \
                           - all_losses.unsqueeze(1)
            sample_weights = F.softmax(weighted_log.view([-1, self.n_gmm * self.n_learners]), dim=1).view(
                [-1, self.n_gmm, self.n_learners])

        assert not torch.isnan(sample_weights).any()
        return sample_weights

    def m_step(self, sample_weights, iterator):
        self.autoencoder.model.eval()
        with torch.no_grad():
            data = []
            for (x, y, indices) in iterator:
                x = x.to(self.device).type(torch.float32)
                x_rep = self.autoencoder.model.encode(x)
                data.append(x_rep)

        data = torch.cat(data)

        # sample_weights = sample_weights.unsqueeze(2).transpose(0, 1).to(self.device)

        pi, mu, var = self.gmm.m_step_with_response(data, sample_weights.sum(dim=2).unsqueeze(2))

        self.learners_weights = sample_weights.mean(dim=0)

    def calc_log_prob_y_x_batch(self, x, y):
        """
        gathers losses for all sample in iterator for each learner in ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor (n_samples, n_learners) with losses of all elements of the iterator.dataset

        """
        n_samples = x.size()[0]
        all_losses = torch.zeros(len(self.learners), n_samples).to(self.device)
        for learner_id, learner in enumerate(self.learners):
            all_losses[learner_id] = learner.calc_log_prob_batch(x, y)

        return -all_losses

    def gather_losses(self, iterator):
        """
        gathers losses for all sample in iterator for each learner in ensemble

        :param iterator:
        :type iterator: torch.utils.data.DataLoader
        :return
            tensor (n_learners, n_samples) with losses of all elements of the iterator.dataset

        """
        n_samples = len(iterator.dataset)
        all_losses = torch.zeros(len(self.learners), n_samples).to(self.device)
        for learner_id, learner in enumerate(self.learners):
            all_losses[learner_id] = learner.gather_losses(iterator)

        return all_losses

    def free_memory(self):
        for learner in self.learners:
            learner.free_memory()
        self.autoencoder.free_memory()

    def free_gradients(self):
        for learner in self.learners:
            learner.free_gradients()
        self.autoencoder.free_gradients()

    def fit_ac_batch(self, batch):
        ac = self.autoencoder
        model = ac.model
        x, y, inx = batch
        old_params = ac.get_param_tensor()

        model.train()

        ac.optimizer.zero_grad()

        recon_loss = self.get_reconstruction_loss(x).mean()
        nll_loss = self.get_nll_loss(x).mean()
        loss = self.reconstruction_weight * recon_loss + self.nll_weight * nll_loss
        loss.backward()

        ac.optimizer_step()

        client_update = (ac.get_param_tensor() - old_params)
        return client_update.cpu().numpy()

    def fit_ac_epoch(self, iterator):
        ac = self.autoencoder
        model = ac.model

        model.train()

        global_recon_loss = 0.
        global_nll_loss = 0.
        n_samples = 0

        for x, y, _ in iterator:
            x = x.to(self.device).type(torch.float32)
            y = y.to(self.device)

            n_samples += y.size(0)

            ac.optimizer.zero_grad()

            recon_loss = self.get_reconstruction_loss(x).mean()
            nll_loss = self.get_nll_loss(x).mean()
            loss = self.reconstruction_weight * recon_loss + self.nll_weight * nll_loss

            loss.backward()

            ac.optimizer.step()

            global_recon_loss += recon_loss.detach() * y.size(0)
            global_nll_loss += nll_loss.detach() * y.size(0)

        return global_recon_loss / n_samples, global_nll_loss / n_samples

    def evaluate_ac_iterator(self, iterator):
        ac = self.autoencoder
        model = ac.model

        model.eval()

        global_recon_loss = 0.
        global_nll_loss = 0.
        n_samples = 0

        with torch.no_grad():
            for x, y, idx in iterator:
                x = x.to(self.device).type(torch.float32)
                y = y.to(self.device)

                n_samples += y.size(0)

                recon_losses = self.get_reconstruction_loss(x).detach()
                recon_loss = recon_losses.mean().item()
                nll_losses = self.get_nll_loss(x).detach()
                nll_loss = nll_losses.mean().item()
                loss = self.reconstruction_weight * recon_loss + self.nll_weight * nll_loss

                global_recon_loss += recon_loss * y.size(0)
                global_nll_loss += nll_loss * y.size(0)

        return global_recon_loss / n_samples, global_nll_loss / n_samples

    def fit_ac_epochs(self, iterator, n_epochs):
        old_params = self.autoencoder.get_param_tensor()
        for step in range(n_epochs):
            self.fit_ac_epoch(iterator)

            if self.autoencoder.lr_scheduler is not None:
                self.autoencoder.lr_scheduler.step()

        update = self.autoencoder.get_param_tensor() - old_params
        return update.cpu().numpy()

    def get_reconstruction_loss(self, x):  # get the loss to backward
        model = self.autoencoder.model

        batch_size = x.size(0)

        x_recon = model(x)
        criterion = self.autoencoder.criterion
        # assert x_recon.min() >= 0. and x_recon.max() <= 1.
        recon_loss = criterion(x_recon, x.view(batch_size, -1)).sum(dim=1)
        return recon_loss

    def get_nll_loss(self, x):  # get the loss to backward
        model = self.autoencoder.model
        batch_size = x.size(0)
        x_rep = model.encode(x)

        return -self.gmm.score_samples(x_rep)

    def freeze_classifier(self):
        for learner in self.learners:
            learner.freeze()

    def unfreeze_classifier(self):
        for learner in self.learners:
            learner.unfreeze()

    def save_state(self, path):
        dic = dict()
        idx = 0
        for learner in self.learners:
            dic[idx] = learner.model.state_dict()
            idx += 1
        dic['ac'] = self.autoencoder.model.state_dict()
        dic['gmm'] = self.gmm.get_all_parameter()
        dic['pi'] = self.learners_weights
        torch.save(dic, path)

    def load_state(self, path):
        dic = torch.load(path)
        idx = 0
        for learner in self.learners:
            learner.model.load_state_dict(dic[idx])
            idx += 1
        self.autoencoder.model.load_state_dict(dic['ac'])
        pi, mu, var = dic['gmm']
        self.gmm.update_parameter(_pi=pi, mu=mu, var=var)
        self.learners_weights = dic['pi']

    def __iter__(self):
        return LearnersEnsembleIterator(self)

    def __len__(self):
        return len(self.learners)

    def __getitem__(self, idx):
        return self.learners[idx]

    # ========================================
    # Communication Compression Support
    # ========================================
    
    def enable_compression(self, args):
        """
        启用通信压缩功能
        
        Args:
            args: 包含压缩配置的参数对象
        """
        from utils.compression import create_compressor
        
        self.compression_enabled = hasattr(args, 'use_dgc') and args.use_dgc
        self.compression_args = args if self.compression_enabled else None
        self.compressor = create_compressor(args) if self.compression_enabled else None
        self.current_round = 0
        
        # 初始化压缩统计
        self.compression_stats = {
            'total_rounds': 0,
            'compressed_rounds': 0,
            'full_upload_rounds': 0,
            'total_compression_ratio': 0.0
        }
        
        if self.compression_enabled:
            print(f"[COMPRESSION] Compression enabled: Top-{args.topk_ratio:.1%} {args.topk_strategy} strategy")
            print(f"   Warmup rounds: {args.warmup_rounds}")
            print(f"   Force upload every: {args.force_upload_every} rounds")
    
    def get_flat_model_params(self) -> torch.Tensor:
        """
        展平所有学习器的模型参数为单个向量
        
        Returns:
            flat_params: 展平的参数张量 (shape: [n_learners * model_dim])
        """
        all_params = []
        for learner in self.learners:
            learner_params = learner.get_param_tensor()
            all_params.append(learner_params)
        
        return torch.cat(all_params, dim=0)
    
    def set_flat_model_params(self, flat_params: torch.Tensor):
        """
        将展平的参数重新分配给各个学习器
        
        Args:
            flat_params: 展平的参数张量
        """
        start_idx = 0
        for learner in self.learners:
            param_size = learner.model_dim
            learner_params = flat_params[start_idx:start_idx + param_size]
            
            # 将参数写回模型
            param_idx = 0
            for param in learner.model.parameters():
                param_numel = param.numel()
                param.data = learner_params[param_idx:param_idx + param_numel].reshape(param.shape)
                param_idx += param_numel
                
            start_idx += param_size
    
    def get_compressed_params(self, current_round: int) -> dict:
        """
        获取压缩后的参数更新
        
        Args:
            current_round: 当前训练轮次
            
        Returns:
            压缩结果字典，包含压缩数据或完整数据
        """
        from utils.compression import should_compress, should_reset_residual
        
        # 更新轮次
        self.current_round = current_round
        
        if not self.compression_enabled:
            # 未启用压缩，返回完整参数更新
            return {
                'type': 'full',
                'data': None,  # 在fit_epochs中处理
                'compressed': False,
                'round': current_round
            }
        
        # 判断是否需要重置残差
        if should_reset_residual(current_round, self.compression_args):
            self.compressor.reset_residual()
            print(f"[COMPRESSION] Round {current_round}: Reset residual cache (force upload)")
        
        # 判断是否压缩
        compress_this_round = should_compress(current_round, self.compression_args)
        
        return {
            'type': 'compressed' if compress_this_round else 'full',
            'data': None,  # 在fit_epochs中填充实际数据
            'compressed': compress_this_round,
            'round': current_round
        }
    
    def apply_compression_to_updates(self, client_updates: torch.Tensor, 
                                   compression_info: dict) -> dict:
        """
        对客户端更新应用压缩
        
        Args:
            client_updates: 客户端参数更新
            compression_info: 压缩配置信息
            
        Returns:
            压缩结果
        """
        if not compression_info['compressed']:
            # 不压缩，返回完整更新
            self.compression_stats['full_upload_rounds'] += 1
            return {
                'type': 'full',
                'data': client_updates.cpu().numpy(),
                'compressed': False,
                'round': compression_info['round']
            }
        
        # 应用残差补偿
        compensated_updates = self.compressor.get_residual_compensated_params(client_updates)
        
        # 执行压缩
        compressed_values, indices, shapes = self.compressor.compress(compensated_updates)
        
        # 解压缩用于残差计算
        decompressed_updates = self.compressor.decompress(compressed_values, indices, shapes)
        
        # 更新残差缓存
        self.compressor.update_residual(compensated_updates, decompressed_updates)
        
        # 更新统计信息
        self.compression_stats['compressed_rounds'] += 1
        self.compression_stats['total_compression_ratio'] += self.compressor.get_compression_ratio()
        
        compression_ratio = self.compressor.get_compression_ratio()
        print(f"[COMPRESSION] Round {compression_info['round']}: Compressed to {compression_ratio:.1%} "
              f"({self.compressor.compressed_size}/{self.compressor.original_size})")
        
        return {
            'type': 'compressed',
            'compressed_values': compressed_values.cpu().numpy(),
            'indices': indices.cpu().numpy(),
            'shapes': shapes.cpu().numpy(),
            'compressed': True,
            'round': compression_info['round'],
            'compression_ratio': compression_ratio
        }
    
    def set_compressed_params(self, compressed_data: dict):
        """
        设置从压缩数据恢复的参数
        
        Args:
            compressed_data: 压缩数据字典
        """
        if compressed_data['type'] == 'full':
            # 完整数据，无需解压缩
            return torch.tensor(compressed_data['data'])
        
        elif compressed_data['type'] == 'compressed':
            # 压缩数据，需要解压缩
            compressed_values = torch.tensor(compressed_data['compressed_values'])
            indices = torch.tensor(compressed_data['indices'])
            shapes = torch.tensor(compressed_data['shapes'])
            
            # 解压缩
            decompressed_params = self.compressor.decompress(compressed_values, indices, shapes)
            return decompressed_params
        
        else:
            raise ValueError(f"Unknown compressed data type: {compressed_data['type']}")
    
    def increment_round(self):
        """
        递增轮次计数器并更新统计信息
        """
        self.current_round += 1
        self.compression_stats['total_rounds'] += 1
    
    def get_compression_stats(self) -> dict:
        """
        获取压缩统计信息
        
        Returns:
            压缩统计字典
        """
        if not self.compression_enabled:
            return {'compression_enabled': False}
        
        stats = self.compression_stats.copy()
        stats['compression_enabled'] = True
        
        if stats['compressed_rounds'] > 0:
            stats['avg_compression_ratio'] = stats['total_compression_ratio'] / stats['compressed_rounds']
        else:
            stats['avg_compression_ratio'] = 1.0
            
        if self.compressor:
            stats.update(self.compressor.get_stats())
            
        return stats
    
    def fit_epochs_with_compression(self, iterator, n_epochs, weights=None, current_round=None):
        """
        带压缩功能的训练方法
        
        Args:
            iterator: 数据迭代器
            n_epochs: 训练轮数
            weights: 样本权重
            current_round: 当前轮次
            
        Returns:
            压缩后的客户端更新或压缩信息字典
        """
        # 执行标准训练
        client_updates = self.fit_epochs(iterator, n_epochs, weights)
        
        # 如果未启用压缩或未提供轮次信息，返回原始更新
        if not self.compression_enabled or current_round is None:
            return client_updates
        
        # 获取压缩配置
        compression_info = self.get_compressed_params(current_round)
        
        # 转换为张量进行压缩处理
        client_updates_tensor = torch.tensor(client_updates)
        
        # 应用压缩
        compressed_result = self.apply_compression_to_updates(client_updates_tensor, compression_info)
        
        return compressed_result


class RepDataset(Dataset):
    """
        Attributes
        ----------
        indices: iterable of integers
        transform
        data
        targets

        Methods
        -------
        __init__
        __len__
        __getitem__
        """

    def __init__(self, data, targets, transform=None):
        if data is None or targets is None:
            raise ValueError('invalid data or targets')
        self.data, self.targets = data, targets

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        return img, target, index
