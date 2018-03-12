import importlib

env_settings = {
    'A2C':dict(agent_class=importlib.import_module('algorithms.A2C'),
               env_name='MountainCar',
               seed=2,
               training_params = {'min_batch_size':500,
                                  'total_timesteps':10000000,
                                  'desired_kl':2e-3},
               algorithm_params = {'gamma':0.97, 
                                   'learning_rate':1e-4}               
               ),
}
