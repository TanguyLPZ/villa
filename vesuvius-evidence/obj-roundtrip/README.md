# Papyrus mesh boundaries before and after OBJ reimport

The baseline and patched Windows converters reimported the same public OBJ meshes at the original tifxyz sampling density. We left the OBJ vertices and UVs as exported. [Build and test log](https://github.com/TanguyLPZ/villa/actions/runs/45061133227)

```powershell
vc_tifxyz2obj w035.tifxyz w035.obj
vc_obj2tifxyz w035.obj w035-roundtrip.tifxyz --tifxyz-source=w035.tifxyz
```

| Segment | Points before | Points after | Quads recovered | Triangles preserved |
| --- | ---: | ---: | ---: | ---: |
| [w035](w035-contour.png) | 62,913 | 63,499 | 584 | 125,826 |
| [w047](w047-contour.png) | 22,435 | 22,886 | 449 | 44,870 |
| [w048](w048-contour.png) | 23,158 | 23,623 | 463 | 46,316 |
| Total | 108,506 | 110,008 | 1,496 | 217,012 |

Red marks boundary samples lost by the baseline conversion. Green marks the recovered samples. The comparisons visualize the same boundary recovery measured in the native Windows tests.

[Public PHerc. 139 w035 data](https://scrollprize.org/tutorial5) · [Public PHerc. 343 w047 and w048 meshes](https://github.com/Nieuwlaar/pherc343-first-letters/tree/e8370f1f88a211166056df62ea6fcfffe4a1063e/outputs)
