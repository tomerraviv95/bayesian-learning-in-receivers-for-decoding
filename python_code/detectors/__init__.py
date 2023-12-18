from python_code.detectors.deepsic.bayesian_deepsic.bayesian_deep_sic_trainer import BayesianDeepSICTrainer
from python_code.detectors.deepsic.end_to_end_deepsic import EndToEndDeepSICTrainer
from python_code.detectors.deepsic.modular_bayesian_deepsic.modular_bayesian_deep_sic_trainer import \
    ModularBayesianDeepSICTrainer
from python_code.detectors.deepsic.seq_deepsic.seq_deep_sic_trainer import SeqDeepSICTrainer
from python_code.utils.constants import DetectorType

DETECTORS_TYPE_DICT = {DetectorType.seq_model.name: SeqDeepSICTrainer,
                       DetectorType.end_to_end.name: EndToEndDeepSICTrainer,
                       DetectorType.model_based_bayesian.name: ModularBayesianDeepSICTrainer,
                       DetectorType.bayesian.name: BayesianDeepSICTrainer}
