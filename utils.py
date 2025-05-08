class EarlyStopper:

    def __init__(self, patience: int = 10, delta: float = 0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = float("inf")
        self.counter = 0
        self.stop = False

    def __call__(self, score: float) -> bool:
        """
        Update with the new metric value; return True if training should stop.
        """
        if score < self.best_score - self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop