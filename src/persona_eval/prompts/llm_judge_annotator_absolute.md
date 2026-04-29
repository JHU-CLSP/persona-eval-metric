###Task Description:
You are evaluating a summary from the perspective of a specific annotator. Consider their background and information needs when assessing quality.
1. Write a detailed feedback that assesses strictly whether the summary best addresses the annotator's query, not the fidelity or the style.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, or explanations.

###Introduction
Automatic summarization tools condense large amounts of text into shorter versions, highlighting the most important points.
But "important" is subjective — what matters to one reader might not matter to another.
This study explores what individual readers actually want from summaries of scientific papers, and how we can measure whether a summary was useful to them.
**We're focused on whether the summary answers your query, not the style or fidelity of the summary.**

###Annotator profile:
- Role: {role}
- Domain: {domain}
- Information needs: {info_needs}

###Query:
{query}

###Response to evaluate:
{summary}

###Score Rubric:
{rubric}
