class AutoScaler():
    def __init__(self, scaling_name: str=None, interval: float=None, save_model_interval: int=None, duration: float=None, start_with_specific_pods_no=None, is_learn: bool=None, has_dataset: bool=None):
        self.scaling_name = scaling_name
        self.active_flag = True
        self.interval = interval
        self.duration = duration
        self.start_with_specific_pods_no = start_with_specific_pods_no
        self.is_learn = is_learn
        self.save_model_interval = save_model_interval
        self.has_dataset = has_dataset

    def get_scaling_name(self):
        return self.scaling_name

    def set_scaling_name(self, scaling_name):
        self.scaling_name = scaling_name

    def get_active_flag(self):
        return self.active_flag

    def set_active_flag(self, active_flag):
        self.active_flag = active_flag

    def get_interval(self):
        return self.interval

    def set_interval(self, interval):
        self.interval = interval

    def get_save_model_interval(self):
        return self.save_model_interval

    def set_save_model_interval(self, save_model_interval):
        self.save_model_interval = save_model_interval

    def get_duration(self):
        return self.duration

    def set_duration(self, duration):
        self.duration = duration
    
    def get_start_with_specific_pods_no(self):
        return self.start_with_specific_pods_no

    def set_start_with_specific_pods_no(self, start_with_specific_pods_no):
        self.start_with_specific_pods_no = start_with_specific_pods_no
    
    def get_is_learn(self):
        return self.is_learn

    def set_is_learn(self, is_learn):
        self.is_learn = is_learn

    def get_has_dataset(self):
        return self.has_dataset

    def set_has_dataset(self, has_dataset):
        self.has_dataset = has_dataset
