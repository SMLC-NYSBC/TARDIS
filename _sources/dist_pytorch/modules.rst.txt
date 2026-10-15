=============
DIST -> Model
=============

DIST layer wrapper
==================

.. automodule:: tardis_em.dist_pytorch.model.layers


DIST graph update-modules
=========================

Collection of all modules wrapped around 'torch.nn.Module' used in the DIST model.

.. automodule:: tardis_em.dist_pytorch.model.modules


Feature embedding
==================

Collection of classes used for Node and Edge embedding.

* **Node embedding** is composed of RGB value or optionally flattened image patches.
	The node embedding use only 'nn.Linear' to embedding (n) dimensional feature object.
	And output [Batch x Feature Length x Channels]
* **Edge embedding** is composed directly from the (n)D coordinate values, where n 
	is av dimension.
	The edge embedding computes 'cdist' operation on coordinate features and produces
	a distance matrix for all points in the given patch. The distance matrix is then
	normalized with an exponential function optimized with the sigma parameter. This
	exponential function normalize distance matrix by putting higher weight on 
	the lower distance value (threshold with sigma). This allows the network to 
	embed distance preserving SO(n) invariance for translation and rotation.

.. automodule:: tardis_em.dist_pytorch.model.embedding
