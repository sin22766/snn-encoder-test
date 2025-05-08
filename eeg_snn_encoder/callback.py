from lightning.pytorch.callbacks import Callback

class TrackBestMetric(Callback):
    """
    Callback to track the best values of the model.
    """

    def __init__(self, monitor='val_loss', mode='min'):
        super().__init__()
        self.metric = monitor
        self.mode = mode
        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def on_validation_end(self, trainer, pl_module):
        current_value = trainer.callback_metrics[self.metric].item()
        if current_value is None:
            print(f"Warning: {self.metric} not found in callback metrics.")
            return
        
        if self.mode == 'min' and current_value < self.best_value:
            print(f"New best {self.metric}: {current_value} | Old best: {self.best_value}")
            self.best_value = current_value
        elif self.mode == 'max' and current_value > self.best_value:
            print(f"New best {self.metric}: {current_value} | Old best: {self.best_value}")
            self.best_value = current_value
    
    @property
    def best_metric(self):
        """
        Returns the best metric value.
        """
        return self.best_value