"""Recovery debounce for simulation input, independent of hardware settings."""
class RecoveryGate:
    def __init__(self, required=3):
        self.required=required;self.good=0;self.holding=True;self.recoveries=0;self.faults=0
    def update(self, healthy):
        if not healthy:
            if not self.holding:self.faults+=1
            self.good=0;self.holding=True
        else:
            self.good=min(self.required,self.good+1)
            if self.holding and self.good==self.required:
                self.holding=False;self.recoveries+=1
        return not self.holding
