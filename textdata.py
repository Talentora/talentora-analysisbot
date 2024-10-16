from summarize import dialogue_processing


#get json type text data first

#transform text data into dialogue format
questions = ["Thank you for joining us today, Seungmin. Can you start by telling me a little about yourself?","That sounds great! Can you share an example of a challenging project you’ve worked on, and how you approached it?",
             "Interesting! How do you typically handle tight deadlines or high-pressure situations?", "That’s impressive. Could you describe a time when you worked with a cross-functional team, and how you handled different perspectives?",
             "Great teamwork! Finally, why are you interested in this role at our company?"]
raw_answers = ["Of course! I have a background in software development, with a focus on machine learning and backend systems. I recently held the position of VP of Tech, where I led multiple projects that integrated AI and real-time data processing, along with improving user experience for various applications. I'm passionate about solving complex problems and leveraging data to drive innovation.","One project that comes to mind was building a Scrabble-style game with real-time multiplayer functionality. The challenge was to design an efficient backend that could handle simultaneous moves from multiple players, keep track of points and board states, and integrate user inputs through a responsive frontend. I designed the game logic using an event-driven architecture, which improved the performance and user experience significantly.",
              "I’ve found that staying organized and maintaining clear communication is key. When under pressure, I break down tasks into manageable pieces and prioritize based on impact. I also make sure to keep stakeholders updated on progress. In one case, we had an urgent client request with a tight deadline, so I coordinated with my team, assigned clear roles, and managed to deliver the project on time without sacrificing quality.",
              "Sure! In a recent project, I collaborated with designers, data scientists, and product managers to develop a recommendation engine for personalized content. Each team had different priorities—the designers focused on user interface, while the data scientists were more concerned with model accuracy. My role was to bridge these gaps, ensuring that the technical implementation met user experience goals without compromising performance. I made sure everyone had a platform to express their concerns and ideas, and we were able to build a product that satisfied both sides.",
              "I’ve always admired your company's commitment to innovation, particularly in AI-driven solutions. The opportunity to work here aligns perfectly with my experience and passion for advancing technology through machine learning and backend architecture. I’m excited about the prospect of contributing to your team and helping to build impactful, scalable products."]

#store the data in pandas DataFrame
d = {'question':questions,'raw-answer':raw_answers}
dialogue = [{'question': q, 'raw-answer': a} for q, a in zip(questions, raw_answers)]

dialogue_processing(dialogue)

# Print the results
for entry in dialogue:
    print(f"Question: {entry['question']}")
    print(f"Raw Answer: {entry['raw-answer']}")
    print(f"Summarized Answer: {entry['summarized-answer']}\n")
    print('---')


#fake transcript
#try summarization with 