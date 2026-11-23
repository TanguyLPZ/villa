# PHerc. 139 withheld-strip repair

This CPU experiment reconstructs a strip between two mapped boundaries on the same papyrus surface. The CT variants use a cubic Hermite bridge as a search surface and score offsets by similarity between raw tomographic profiles. The connected variants constrain displacement changes along each row and favor agreement across rows. A geometry variant transfers local relief from triangles on known neighboring windings when the normal intersects those triangles near the expected separation. It leaves the cubic bridge in place where guiding geometry is missing.

Run from this directory with Python 3.12.

```sh
python -m pip install numpy==2.2.6 scipy==1.15.3 tifffile==2025.5.10 matplotlib==3.10.3
python -m unittest -v
python benchmark.py --inputs inputs --out evidence
python benchmark.py --inputs inputs --out holdout --split holdout
```

The input meshes and CT voxels come from the [Vesuvius Challenge public data](https://scrollprize.org/data) for PHerc. 139. The script downloads the four reference meshes and only the needed chunks of the level-zero 2.399 µm scan. It leaves the input mesh positions unchanged outside each masked strip.

The benchmark scores linear interpolation, two cubic interpolations, independent CT profile matching, two connected CT variants, and the neighboring-winding guide at three gap widths. The second connected method scores similarity between adjacent CT profiles at one third of a mesh step, with a smooth displacement prior. The six development regions are on w030 and w031. Six regions on two other windings, w029 and w032, test the follow-up with locations chosen independently of repair error. The masked target geometry and winding labels are held out from fitting. The absolute normal error uses the corresponding reference vertex and local reference normal. A vertex counts as having a different nearest winding if a vertex of another published winding among w029–w032 is closer. This is a proxy based on incomplete reference surfaces. Pooled counts include repeated vertices at nested gap widths.

The example is fixed to the first region in each split with the widest gap. The image shows one curved CT section and the projected reference trace. The evidence directory includes that example's TIFXYZ meshes and the measurements for all cases. It is possible to inspect the example against the same parent volume in VC3D.
