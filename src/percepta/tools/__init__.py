"""Built-in perception tools (spec §5.4).

Every module here must be *import-safe without GPU, weights or network*: heavy imports
(``torch``, ``sam3``, model checkpoints) belong inside ``load()``, because
:func:`percepta.registry.ToolRegistry.discover` imports each module to find its class.
"""
