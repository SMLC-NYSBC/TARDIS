# DIST -> DataLoader
## General DataLoader
General class for creating Datasets. It works by detecting all specified 
file formats in the given directory and return the index list.

```{eval-rst}
.. autoclass:: tardis_em.dist_pytorch.datasets.dataloader.BasicDataset
```

## Specialized DataLoader's
### Filament structure DataSet
```{eval-rst}
.. autoclass:: tardis_em.dist_pytorch.datasets.dataloader.FilamentDataset
```

### PartNet synthetic DataSet
```{eval-rst}
.. autoclass:: tardis_em.dist_pytorch.datasets.dataloader.PartnetDataset
```

### ScanNet V2 synthetic DataSet
```{eval-rst}
.. autoclass:: tardis_em.dist_pytorch.datasets.dataloader.ScannetDataset
```

### ScanNet V2 with RGB values synthetic DataSet
```{eval-rst}
.. autoclass:: tardis_em.dist_pytorch.datasets.dataloader.ScannetColorDataset
```

### Stanford S3DIS DataSet
```{eval-rst}
.. autoclass:: tardis_em.dist_pytorch.datasets.dataloader.Stanford3DDataset
```

## Helper Functions
### Build dataloader
```{eval-rst}
.. autofunction:: tardis_em.dist_pytorch.datasets.dataloader.build_dataset
```

### Augmentation
```{eval-rst}
.. automodule:: tardis_em.dist_pytorch.datasets.augmentation
```