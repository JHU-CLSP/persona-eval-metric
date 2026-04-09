###Task Description:
You are evaluating summaries from the perspective of a specific annotator. Consider their background and information needs when assessing quality.
1. Write a detailed feedback that compares the two responses strictly whether the summary the summary best addresses your query, not the fidelity or the style.
2. After writing a feedback, choose which response best addresses your query. If Response A is better, write "A". If Response B is better, write "B".
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, or explanations.

###Introduction
Automatic summarization tools condense large amounts of text into shorter versions, highlighting the most important points. 
But "important" is subjective — what matters to one reader might not matter to another.
This study explores what individual readers actually want from summaries of scientific papers, and how we can measure whether a summary was useful to them. </p>
**We're focused on whether the summaries answer your query, not the style or fidelity of the summaries.**

###Annotator profile:
- Role: {role}
- Domain: {domain}
- Information needs: {info_needs}

###Query:
{query}

###Source Document:
{source}

###Response A:
{summary_a}

###Response B:
{summary_b}