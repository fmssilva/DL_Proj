What metric does each phase use, and why
Phase 1 — Training loop (train_one_epoch)
Uses: val_loss (CrossEntropyLoss)

The gradient descent optimises cross-entropy loss. That's the only thing that can be backpropagated — F1 is not differentiable.

Phase 2 — Early stopping (EarlyStopping)
Uses: val_loss — NOT macro-F1

The checkpoint that gets saved is the one with the lowest validation loss, not the highest val_macro_F1.

This is a deliberate trade-off. Val loss is smoother and more reliable per-epoch (especially with few data). Val macro-F1 is noisy on small validation sets — a single batch flip can swing it ±0.02. Using loss for checkpointing is the standard practice.

Consequence to be aware of: the "best" checkpoint might not be the epoch with the highest F1. On the full run this matters less because loss and F1 are generally correlated, but it's worth knowing.

Phase 3 — Experiment selection (results_tracker / best_name)
Uses: val_macro_f1 — the competition metric

After all experiments finish, we pick the winner by macro-F1 — exactly the Kaggle competition metric. This is correct.

Phase 4 — Reporting and submission
Uses: val_macro_f1 and per_class_f1

The final classification report, confusion matrix, and JSON all report macro-F1. The submission CSV is generated from the best macro-F1 experiment's checkpoint.

Summary table
Phase	What drives it	Why
Gradient update	CrossEntropyLoss	only differentiable option
Checkpoint save	lowest val_loss	smoother signal, less noisy per-epoch
Best experiment pick	highest val_macro_f1	competition metric
Report & submission	val_macro_f1 + per-class F1	competition metric
Should we change early stopping to use macro-F1 instead of loss?
It's a reasonable question. With the full dataset (2880 train / 720 val) the val F1 is stable enough to use directly. The change would be:

Verdict for now: keep val_loss for early stopping. Reason — with only 30 epochs and a clear overfitting problem, the loss and F1 curves are well correlated. Switching to F1-based stopping is a micro-optimisation that won't move the needle vs the bigger improvements (architecture, regularisation). Worth noting in the report as a known design choice.













time:
spent some 10 minnuts running fast runs to confirm the code works with no errors... with cpu and gpu... 
then 15 minutes running full and getting best modeel from first bets

then more exploration around thebest model to confirm if we can get better 


i ran
cheeck the data and complete the report file: C:\Users\franc\OneDrive\Nossa_Pasta_2\5. Universidade\Cadeiras\DL\DL_Proj\assignment_1\task1\TASK1_REPORT.md

and there tell me the TODOs i should do to update the notebook if needed...


and then nalyse the best model we have. what is it? 
can we improve further on it? 
example if bottleneck worked... so lets explore more configurations...
or other "variations" of the best model to make it better?

and in terms of epochs all the models stoped earlier or maybe ssome still needed more training? 



and in terms of range of values, explain me what is our goal? to have scores = 0 or big numbers? our to read each core, or loss or f1...?? 

and in terms of general omparison with some MLP models for similar problems are we scoring good values or is there maybe somethign wrong? 

this evolution in the plot is also good? how to read it? (pasted image)
