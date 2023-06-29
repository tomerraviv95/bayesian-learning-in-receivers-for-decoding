from python_code.detectors.deepsic.bayesian_deepsic.bayesian_deep_sic_trainer import BayesianDeepSICTrainer
from python_code.detectors.deepsic.model_based_bayesian_deepsic.model_based_bayesian_deep_sic_trainer import \
    ModelBasedBayesianDeepSICTrainer
from python_code.detectors.deepsic.seq_deepsic.seq_deep_sic_trainer import SeqDeepSICTrainer
from python_code.detectors.dnn.bayesian_dnn.bayesian_dnn_trainer import BayesianDNNTrainer
from python_code.detectors.dnn.dnn_trainer import DNNTrainer
from python_code.utils.constants import DetectorType

DETECTORS_TYPE_DICT = {DetectorType.seq_model.name: SeqDeepSICTrainer,
                       DetectorType.model_based_bayesian.name: ModelBasedBayesianDeepSICTrainer,
                       DetectorType.bayesian.name: BayesianDeepSICTrainer,
                       DetectorType.black_box.name: DNNTrainer,
                       DetectorType.bayesian_black_box.name: BayesianDNNTrainer}
