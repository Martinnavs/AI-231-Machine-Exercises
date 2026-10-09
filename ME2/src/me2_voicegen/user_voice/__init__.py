"""User-voice weak-phrase augmentation (feature `user-voice-weak-phrases`).

The user's own recordings of the grammar's "weak" phrases, voice-converted into a
random stratified sample of the fil50 persona voices and appended (train split only)
to the VCM datasets, then perturbed exactly like the rest of the dataset (ESC-50 noisy
siblings at the fil50 rate + ambient babble overlay via the existing tool) and trained
with the unchanged recipe. Stages: prep -> plan -> convert -> (QA via accent_balance.qa)
-> collate -> (ambient mix via dataset_tools.mix_ambient_noise) -> assemble.
"""
