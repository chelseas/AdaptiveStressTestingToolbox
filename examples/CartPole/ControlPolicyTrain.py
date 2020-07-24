from CartPole.cartpole import CartPoleEnv
from garage import wrap_experiment
from garage.envs import GarageEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.algos import TRPO
from garage.tf.policies import CategoricalMLPPolicy

@wrap_experiment
def train_cartpole_policy(ctxt=None, seed=1):
    """Train TRPO with CartPole-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
    set_seed(seed)
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        env = GarageEnv(CartPoleEnv(use_seed=False))

        policy = CategoricalMLPPolicy(
            name='protagonist',
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(32, 32)
        )

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            max_kl_step=0.01,
        )

        runner.setup(algo, env)
        runner.train(n_epochs=120, batch_size=4000)


train_cartpole_policy()