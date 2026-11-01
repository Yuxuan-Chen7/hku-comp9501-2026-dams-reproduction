# Data releases used

All hashes below are locally recorded fingerprints of the inspected releases, not publisher-provided checksums. Downloads and preexisting archives are verified against them. The repository does not redistribute raw messages/features.

| File | Compressed bytes | Source |
| --- | ---: | --- |
| `20news-bydate-matlab.tgz` | 7,575,962 | [20 Newsgroups count-matrix archive](http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz) |
| `data_task_a.zip` | 7,560,626 | [Archived original ECML/PKDD 2006 release](https://web.archive.org/web/20060429084857id_/http://www.ecmlpkdd2006.org:80/data_task_a.zip) |
| `evaluation_data_labeled.zip` | 4,459,242 | [Archived official evaluation labels](https://web.archive.org/web/20060617074214id_/http://www.ecmlpkdd2006.org:80/evaluation_data_labeled.zip) |

Total archive size: **19,595,830 bytes** (about 19.6 MB). Prepared sparse matrices, dense kernels in RAM, cached predictions and model files require additional space.

```text
20news-bydate-matlab.tgz
086fdc442ae2c269c53c6383f641ff92d55cb2b77da19d2cf3562473fc73dd19

data_task_a.zip
916e4d07ba526391bab3a8d2e873cd8e3ebae838203c53d3626bf67e2e96cdfc

evaluation_data_labeled.zip
dda675d372a516e3fd1daeb893a41b9542c904593dc02c87a9d18ae7da49dd7f
```

20NG combines the count release's train/test portions before domain construction, as required by the paper's source/target task definitions. There are 18,774 documents and 61,188 feature IDs. No new vocabulary, TF-IDF weighting or test-label feature selection is performed. L2 row normalization is an explicit implementation choice.

The Email Spam release contains several task/tuning files; only Task A's public labeled training file and U00/U01/U02 evaluation feature rows with subsequently released labels are used. The combined matrix has 11,500 rows and 206,908 feature IDs. L2 normalization is applied. The source features must exactly match the corresponding originally unlabeled rows. Tuning files are excluded.

The TRECVID experiment needs the original channel assignments, video feature vectors and concept labels used by the paper. The [Columbia374 official page](https://www.ee.columbia.edu/ln/dvmm/columbia374/) describes the historical release, but the download process failed during this project. No substitute video dataset is used, and no local TRECVID result is claimed.
