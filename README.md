Panorama Stitching
==================

Run codes
----
Install packages:
```shell
pip install requirements.txt
```

Run in specific dataset e.g. "data1":
```shell
python main.py --task 1
```

Run in specific descriptor and matcher e.g. `SIFT` descriptor and `BF` matcher:
```shell
python main.py --des SIFT --matcher BF
```

Note
----
You may need to change the `BASE_DIR` in `main.py`, to specify input directory.