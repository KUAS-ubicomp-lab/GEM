# <B> <I> GEM </I> </B>
<h2><u>G</u>raph Attention <u>E</u>ncoder for <u>M</u>ulti-task Depression Severity Detection in Multi-party Conversations.</h2>

# System Design 
![GEM High-level Design drawio](https://github.com/user-attachments/assets/8527d839-55ae-4cda-9238-8434bd316a51)

# Approach
Multi-party conversations (MPCs) consists of complex discourse-level structures among multiple interlocutors and respective utterances. Compared to two-party conversations, MPCs significantly enhances the discourse-base natural language understanding allowing to model specific downstream tasks including emotion recognition of the interlocutors. Recent advancements of graph attention networks (GATs) have been utilized to discover various downstream tasks on MPC modelling using the root-level and sub-level utterance structures. <i> GEM, </i> incorporates these root-level and sub-level MPC utterances to model depression detection and severity classification downstream tasks. To our knowledge, we make the first attempt to utilize root-sub relationships between root-level and sub-level utterances in an MPC to develop a GAT-based encoder for depression detection.

# Downstream Tasks Formation
Three downstream tasks, Depressed Utterance Detection (DUD), Depressed Interlocutor Recognition (DIR), and Depression Severity Classification (DSC), are formed based on the outputs of the task-specific layers of <i>GEM</i>.

# Experimental Setup
Requirements:
- Python 3.8
- PyTorch 2.0
- OpenAI

Setting OpenAI API key: `export OPENAI_API_KEY=yourkey`