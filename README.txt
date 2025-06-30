


Training：
1. Revise the TrainConfig and m_dis_type, m_database_path in config.py

2. Run build_dataset.py to generate dataset_out folder.

3. Run main.py to train a model -- python maim.py train

Testing:
1. Revise the TestConfig in config.py 

2. Run main.py to obtain the test results -- python main.py test



data:
img300_kodak: contains 324 images, where the front 300 images are used for training, the last 24 images are Kodak24 dataset, which are used for testing.
Set5: contains 5 images.
Set12: contains 12 images.

XXX_A11_3 -- XXX dataset encrypted by MBSE algorithm with HE level
XXX_A11_15 -- XXX dataset encrypted by MBSE algorithm with ME level
XXX_A11_25 -- XXX dataset encrypted by MBSE algorithm with SE level
XXX_A07_3 -- XXX dataset encrypted by RISE algorithm with SE level
XXX_A07_4 -- XXX dataset encrypted by RISE algorithm with ME level
XXX_A07_5 -- XXX dataset encrypted by RISE algorithm with HE level
XXX_A22_1 -- XXX dataset encrypted by GLSE algorithm with HE level
XXX_A22_2 -- XXX dataset encrypted by GLSE algorithm with ME level
XXX_A22_3 -- XXX dataset encrypted by GLSE algorithm with SE level







