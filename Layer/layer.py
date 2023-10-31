from abc import abstractmethod
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None
        raise NotImplementedError
    @abstractmethod
    def get_input(self):
        return self.input
    
    @abstractmethod
    def get_output(self):
        return self.output
    
    @abstractmethod
    def get_input_shape(self):
        return self.input_shape
    
    @abstractmethod
    def get_output_shape(self):
        return self.output_shape
    
    @abstractmethod
    def forward_propagation(self,input):
        raise NotImplementedError
    
    @abstractmethod
    def backward_propagation(self,output_error, learning_rate):
        raise NotImplementedError
    