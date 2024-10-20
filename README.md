 # Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) [![NeurIPS 2024 (Spotlight)](https://img.shields.io/badge/NeurIPS'24-blue)](https://neurips.cc/)

Official implementation for "Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts" accepted by NeurIPS 2024 (Spotlight).  

- Authors: [Sukwon Yun](https://sukwonyun.github.io/), [Inyoung Choi](https://www.linkedin.com/in/inyoung-choi-77105221b/), [Jie Peng](https://openreview.net/profile?id=~Jie_Peng4), Yangfan Wu, [Jingxuan Bao](https://jingxuanbao.github.io/), Qiyiwen Zhang, [Jiayi Xin](https://www.linkedin.com/in/jiayi-xin/), [Qi Long](https://www.med.upenn.edu/long-lab/currentgroupmembers.html) and [Tianlong Chen](https://tianlong-chen.github.io/)

## Overview

Multimodal learning has gained increasing importance across various fields, offering the ability to integrate data from diverse sources such as images, text, and personalized records, which are frequently observed in medical domains. However, in scenarios where some modalities are missing, many existing frameworks struggle to accommodate arbitrary modality combinations, often relying heavily on a single modality or complete data. This oversight of potential modality combinations limits their applicability in real-world situations. To address this challenge, we propose Flex-MoE (Flexible Mixture-of-Experts), a new framework designed to flexibly incorporate arbitrary modality combinations while maintaining robustness to missing data. The core idea of Flex-MoE is to first address missing modalities using a new missing modality bank that integrates observed modality combinations with the corresponding missing ones. This is followed by a uniquely designed Sparse MoE framework. Specifically, Flex-MoE first trains experts using samples with all modalities to inject generalized knowledge through the generalized router ($\mathcal{G}$-Router). The $\mathcal{S}$-Router then specializes in handling fewer modality combinations by assigning the top-1 gate to the expert corresponding to the observed modality combination. We evaluate Flex-MoE on the ADNI dataset, which encompasses four modalities in the Alzheimer's Disease domain, as well as on the MIMIC-IV dataset. The results demonstrate the effectiveness of Flex-MoE highlighting its ability to model arbitrary modality combinations in diverse missing modality scenarios.

<img src="assets/Model.png" width="100%">

#### Full code with ADNI dataset Preprocessing will be uploaded soon (Oct 19th).

