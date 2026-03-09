#!/bin/sh
set -e

rm -f src/smart/model/ego_gmm_smart.py
rm -f src/smart/modules/ego_gmm_agent_decoder.py
rm -f src/smart/modules/ego_gmm_smart_decoder.py
rm -f src/smart/metrics/cross_entropy.py
rm -f src/smart/metrics/ego_nll.py
rm -f src/smart/metrics/gmm_ade.py
rm -f src/smart/metrics/next_token_cls.py
rm -f configs/model/ego_gmm.yaml
rm -f configs/experiment/pre_bc.yaml
rm -f configs/experiment/clsft.yaml
rm -f configs/experiment/local_val.yaml
rm -f configs/experiment/wosac_sub.yaml
rm -f scripts/train.sh
rm -f scripts/local_val.sh
rm -f scripts/wosac_sub.sh

echo "Legacy NTP/GMM files removed."
