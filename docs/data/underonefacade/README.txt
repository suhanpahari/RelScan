UnderOneFacade: A Large-Scale Benchmark for Semantic Facade Segmentation

UnderOneFacade is a large-scale benchmark for semantic segmentation of
urban facade point clouds. It contains high-resolution terrestrial laser
scanning (TLS) and mobile laser scanning (MLS) point
clouds acquired across multiple countries and architectural styles.

The benchmark is designed to evaluate the robustness and generalization
of 3D semantic segmentation methods under realistic conditions,
including:

-   large-scale outdoor environments,
-   severe class imbalance,
-   fine-grained architectural semantics,
-   cross-country domain shifts,
-   varying acquisition modalities and point densities.

The benchmark introduces a hierarchical semantic taxonomy consisting of:

-   LoFG3 – 15 fine-grained facade classes
-   LoFG2 – 5 coarse semantic groups

Highlights

-   Multi-country benchmark (United Kingdom, Germany, Singapore)
-   Diverse architectural styles
-   High-resolution laser scanning data
-   Fine-grained facade annotations
-   Hierarchical semantic labels (LoFG2 & LoFG3)

Dataset Highlights

-   Countries: 3
-   Annotated Points: 2.7BN
-   LoFG3 Classes: 15
-   LoFG2 Classes: 5

Dataset Structure

You can use the overall dataset --> "UnderOneFacade" with the defined splits or use our "UnderOneFacade per Country" to further analyse country-specific architecture impact. 

The data has (x, y, z, label, rgb[for UK/Singapore]):

UnderOneFacade/

UnderOneFacade/

+-- training/
|
-- -...(.txt; .las)
|
+-- validation/
|
-- -...(.txt; .las)
|
+-- test/
|
-- -...(.txt; .las)
|

UnderOneFacade per Country/
|
+-- UK/
|   +-- training/
	-- -...(.txt; .las)
|   +-- validation/
	-- -...(.txt; .las)
|   `-- test/
	-- -...(.txt; .las)
|
+-- Germany/
|   +-- training/
	-- -...(.txt; .las)
|   +-- validation/
	-- -...(.txt; .las)
|   `-- test/
	-- -...(.txt; .las)
|
+-- Singapore/
|   +-- training/
	-- -...(.txt; .las)
|   +-- validation/
	-- -...(.txt; .las)
|   `-- test/
	-- -...(.txt; .las)
|

-- You can also find weights, code, and useful snippets in the other folders with appropriate readme files!

Semantic Classes

LoFG3: Wall, Window, Door, Balcony, Molding, Decoration, Column, Arch,
Stairs, Ground Surface, Terrain, Roof, Blinds, Interior, Other.

LoFG2: - Structural - Opening - Decoration - Floor - Other Elements

Main Benchmark Tasks

-   Semantic Segmentation
-   Cross-country Generalization
-   Hierarchical Semantic Segmentation

Evaluation

Metrics: - Overall Accuracy (OA) - Mean Precision (mP) - Mean Recall
(mR) - Mean F1 - Mean IoU - Per-class F1

Baselines (using the default architecture with identical training budgets as specified in the paper)

-   PointNet++
-   Point Transformer
-   Point Transformer v3
-   DGCNN
-   KPConv
-   OctFormer
-   Superpoint Transformer

Download

Just download from the link! https://drive.google.com/drive/folders/1Yzz7PmyeK1qeOtkTFCfkbw7IEHXcMJo8?usp=sharing

Citation

Please cite the UnderOneFacade paper if you use the dataset.

License

CC BY 4.0 (TL;DR - only cite and use)

Contact

Computer Vision for Digital Twins (CV4DT) University of Cambridge.

Website

https://jiangyuanwangyi.github.io/UnderOneFacade_official/

Acknowledgements

We thank all collaborators and annotators who contributed to the
creation and validation of the benchmark.
