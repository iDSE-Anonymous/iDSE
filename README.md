# iDSE
iDSE is a fast and accurate analytical framework for design space exploration of CPU. 
iDSE uses a novel mechanism, namely iterative refinement training. The key idea is to use a few key design points for initial training, and iteratively predict additional ones to refine training, so as to use less design points and simulation time to fast train accurate analytical models. This idea is inspired by an experimental finding that the key design points are the cornerstones to train accurate analytical models. In addition, a workload-aware sampler is integrated for initial key design points generation. We use two MLPs as the analytical models for CPI and power prediction.
