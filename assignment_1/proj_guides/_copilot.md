                
Before implementing anything, read 				
    \assignment_1\PROJ_PLAN.md			
    and all relevant existing files. Understand the full project structure before touching any code. Think as a software architect first, not a coding agent.			
                
Code Quality				
    Simple, clean, well-organized. No over-engineering.			
    Use what Python/PyTorch/sklearn already give you — don't reimplement what libraries handle well.			
    Small files with single, clear responsibilities. Name everything so the structure is self-documenting.			
    Domain-centered structure (not tests/, plots/ folders — keep related things together).			
    Only implement what is currently needed. No "might be useful later" functions.			
                
Comments & Logs				
    Short comment above each function (one line max).			
    Inline comments during the code itself — casual, natural language, like one dev explaining to another. No formal prose. No emojis anywhere.			
    Logs: only what's needed to pinpoint errors. Single-line, concise, no emojis (terminal encoding issues).			
                
File & Folder Structure				
    Prefer this pattern — everything by domain:			
    src/			
        datasets/		
            dataset.py       # loading + transforms	
            eda.py           # EDA functions	
            eda_plots.py     # EDA visualizations	
            dataset_test.py  # tests for data loading	
    No utils.py dumping ground with 50 mixed functions. If a file starts doing too many unrelated things, split it.			
                
Testing During Development				
    Each file has its own if __name__ == "__main__" block with tests. Run everything locally on CPU with small data samples first. Only move to Colab+GPU once local tests pass.			
    Good tests to include in DL projects:			
        Data tests: shapes correct, labels encoded right, no NaNs, class distribution matches CSV		
        Transform tests: output tensor shape, value range [0,1] or normalized		
        Model tests: forward pass with dummy input gives expected output shape		
        Training tests: loss decreases after 1-2 steps on a tiny batch (sanity check)		
        Submission tests: output CSV has correct number of rows, valid class names, correct format		
                
Before Implementing Any Task				
    Read all relevant existing files — avoid duplication.			
    Think the 3 best options at architecture level (where/how it fits in the project).			
    Think the 3 best options at implementation level (how to write the code locally).			
    Choose the best one — clean, simple, maintainable. No shortcuts, no over-engineering.			
                
Workflow				
    implement → local test (CPU, small data) → fix → notebook (Colab, GPU)			
    Never update the notebook until the Python files are tested and working.			
                
Terminal				
    PowerShell — use ; not && for chaining commands.			
                
On Finishing Each Task				
    Give a short chat summary: what was implemented, in which file/folder, and what the data flow is. No summary files. Just tell me in chat.			
                
Take your time. Quality over speed.				