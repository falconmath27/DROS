# P5 full-grid results

Each scenario rebuilds its reward and refits OSPG-GRU, OSPG-MLP and LSM. The three
rule methods are evaluated on the same windows. Cost scenarios are 0x, 1x and 2x the
configured venue assumptions. Fixed-exit rescoring is not mixed into these results.

The equity constituents and crypto ranking are fixed supplied or retrieval-date
universes. They are not historical point-in-time membership and therefore introduce
survivorship and selection bias.

Entry and exit use adjusted daily-close reference prices plus assumed friction. This
is a stylized execution model and does not claim that an order placed after observing
the close fills at that close.

FX and gold remain optional and are outside the configured core grid.
