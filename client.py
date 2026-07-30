from MLP_model import train_model, test_model
import copy

class HospitalClient:
    def __init__(self, client_id, train_loader, test_loader, model):
        """
        Initializes a Federated Client representing a single hospital.
        
        Args:
            client_id (str): Name of the hospital (e.g., 'Cleveland').
            train_loader: PyTorch DataLoader for local training.
            test_loader: PyTorch DataLoader for local validation.
            model: The PyTorch neural network architecture.
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        
    def set_weights(self, global_weights):
        """
        Overwrites the client's local model weights with the newly 
        aggregated global weights from the server.
        """
        self.model.load_state_dict(copy.deepcopy(global_weights))

    def get_weights(self):
        """
        Retrieves the current local weights to send to the server.
        """
        return self.model.state_dict()

    def train(self):
        """
        Executes the local training loop using the client's private data.
        Returns the updated model weights.
        """
        global_wieghts = train_model(self.model, self.train_loader)
                
        return global_wieghts

    def evaluate(self):
        """
        Tests the client's local model against its local test set.
        Returns accuracy and loss metrics.
        """
        results = test_model(self.model, self.test_loader)
        
        return results