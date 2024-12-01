Yizheng Chen at Mon Nov 18, 2024 8:16pmat Mon Nov 18, 2024 8:16pm
Thank you for your report!

Problem Statement: Good problem statement. In related work, the report mentions that “we proposed that start with LLM then refine the result with feedback from SAST tools.“ The report could elaborate on this more in the problem statement.

Related Work: good list of related work in combing LLM and SAST.

Methodology: Why do you choose pybughive? Could StackOverflow contain answers to the problems in pybughive?
- its easy
- why is it a problem if the dataset is inthe training and stackoverflowm, thats a given at this point

“we need the 2nd stage that call for the assistance from SAST tools, which has relative low false positive rate” Can SAST tools reduce false positives of LLMs? [9] actually used LLM to reduce the false positives of SAST tools.
- what does she mean??

It’s great to start with a simple procedure of LLM first then SAST tool next, but I wonder if there is a better way to integrate LLM to parse the information from SAST tools, or decide how to use SAST tools. One key feature of LLM agents is to have autonomy to decide how others use tools.
- yes, there would be better ways to use SAST tools, but that's too much work

Results: Thanks for developing the framework. The expectation of the “experiment” part is to answer a small research question like what the hacker role does. For example, although your framework does not currently run yet, the report could include a motivating example, e.g., see motivating example in [9]. One would only need to manually prompt LLM to detect vulnerable code, take some potential outputs, and demonstrate how SAST can help with that.
Subtracting 2 points from the results part.
- done

Next Steps: there are different tradeoffs for LLM agents to have access to the Internet verse SAST. Internet may contain potential solutions. SAST analyzer rules usually have many FPs and FNs. At the end of the report, I am confused by what this project is trying to achieve. The Phase I method does not really correspond to the goal stated in the problem statement. Will you only evaluate the result using agent having access to SAST?
“How much better it can do” what would be the baseline you compare against?
- baseline: just use model, no other  tools
- mod1: use StackOverflow
- mod2: use DLint
- mod1+2: use StackOverflow and DLint

Please clearly state what each member of the team contributed to the course project report. The current version of missing that.
- done