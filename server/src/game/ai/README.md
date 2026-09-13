# Neural player

Each server owns exactly one `NeuralPlayerBot`, while all other generated players keep the existing scripted controller. The neural player queries the same viewport used for an ordinary player's network snapshot, compresses a subset of visible information into 29 local features, and produces only normal movement, aim, swing, and sword-throw inputs. It cannot inspect off-screen entities, account data, combat internals, or hidden map state, and it never activates evolution abilities.

The checked-in model is a dependency-free 29→32→24→6 multilayer perceptron. Run `yarn train:neural-bot` from `server/` to reproduce it from deterministic, in-memory imitation scenarios. The trainer enforces a 100 MB final-model limit and a validation quality gate. Its scratch directory is removed after every pass; no generated replay dataset or checkpoint is retained.
