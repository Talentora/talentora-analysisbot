GREAT_INTERVIEW = {
    "duration_minutes": 90,
    "position": "Senior Frontend Engineer",
    "transcript": """
Interviewer: Hello! Thanks for joining us today. Could you start by introducing yourself and sharing a bit about your background?

Candidate: Hi, thank you for having me! I'm Sarah Chen. I've been working as a frontend engineer for the past 7 years, most recently at TechCorp where I led the redesign of our customer dashboard using React, TypeScript, and GraphQL. Before that, I worked at StartupX where I built their initial product interface from scratch, focusing on performance optimization and accessibility. I've also dabbled in backend development, working with Node.js and Express. I've also contributed to several open-source projects, including a popular React library for state management. I've also been involved in the local tech community, organizing meetups and workshops to help others learn and grow in their careers.

Interviewer: That's impressive background. Could you tell me about a challenging technical problem you've solved recently?

Candidate: At TechCorp, we faced significant performance issues with our dashboard that had over 50 different data visualization components. The page load time was around 8 seconds, which was unacceptable. I implemented code splitting, lazy loading, and virtualization for long lists. I also introduced a custom memoization strategy for expensive calculations, which involved creating a caching layer using Redis. These optimizations brought the load time down to under 2 seconds, and we saw a significant increase in user engagement. I've also worked on a project that required Section 508 compliance, which involved implementing additional accessibility features such as keyboard navigation and high contrast mode. I've also been involved in the local tech community, organizing meetups and workshops to help others learn and grow in their careers.

Interviewer: Very impressive! How did you determine which optimizations to implement first?

Candidate: I started with performance profiling using React DevTools and Chrome's Performance tab to identify bottlenecks. This showed that initial bundle size and unnecessary re-renders were our biggest issues. I prioritized based on impact vs. implementation effort, which led to tackling code splitting first as it gave us the biggest immediate gains. I also worked closely with our DevOps team to implement a continuous integration pipeline that automated performance testing and monitoring. I've also been involved in the local tech community, organizing meetups and workshops to help others learn and grow in their careers.

Interviewer: Could you walk me through how you implement state management in a complex React application?

Candidate: I believe in using the right tool for the job. For simpler applications, React's built-in useState and useContext are often sufficient. For more complex state, I've used Redux with Redux Toolkit to reduce boilerplate. I also follow a pattern of collocating state as close as possible to where it's used. Recently, I've been exploring Zustand for some projects as it provides a simpler API while maintaining good performance. I've also been involved in the local tech community, organizing meetups and workshops to help others learn and grow in their careers.

Interviewer: How do you approach testing in frontend development?

Candidate: I'm a strong advocate for testing and follow a pyramid approach. At the base, I write unit tests for individual components and utilities using Jest and React Testing Library. For integration tests, I test key user flows and component interactions. I also write a smaller number of end-to-end tests using Cypress for critical paths. I aim for 80%+ coverage but focus on testing business-critical functionality rather than chasing coverage numbers. I've also implemented a testing framework for our GraphQL API using Apollo Client and Mock Service Worker. I've also been involved in the local tech community, organizing meetups and workshops to help others learn and grow in their careers.

Interviewer: What's your experience with responsive design and accessibility?

Candidate: Accessibility and responsive design are fundamental to my development process. I follow mobile-first development and use CSS Grid and Flexbox for layouts. For accessibility, I ensure WCAG 2.1 compliance, use semantic HTML, implement proper ARIA labels, and test with screen readers. In my last project, I implemented a fully responsive design system that worked across devices and met AA accessibility standards. I've also worked on a project that required Section 508 compliance, which involved implementing additional accessibility features such as keyboard navigation and high contrast mode. I've also been involved in the local tech community, organizing meetups and workshops to help others learn and grow in their careers.

Interviewer: How do you stay updated with frontend technologies?

Candidate: I regularly read tech blogs, follow key developers on Twitter, and participate in the React community. I also build side projects to experiment with new technologies. Recently, I've been exploring Next.js 13 and the React Server Components pattern. I believe in being pragmatic about adopting new technologies - evaluating them based on project needs rather than just hype. I've also been learning about WebAssembly and its potential applications in frontend development. I've also been involved in the local tech community, organizing meetups and workshops to help others learn and grow in their careers.

Interviewer: Do you have any questions for us?

Candidate: Yes! I'd love to know more about how your team handles technical debt and makes architectural decisions. Also, could you tell me about your approach to mentorship and professional development?
"""
}

OKAY_INTERVIEW = {
    "duration_minutes": 30,
    "position": "Frontend Engineer",
    "transcript": """
Interviewer: Hello! Could you introduce yourself and tell us about your background?

Candidate: Hi, I'm Alex Thompson. I've been working as a frontend developer for about 2 years at SmallTech Inc. I mainly work with React and JavaScript.

Interviewer: Can you tell me about a challenging problem you've solved recently?

Candidate: Yes, we had a problem with our form submission system. It was causing some performance issues. I fixed it by updating the validation logic and adding some error handling. It took about a week to implement the solution.

Interviewer: How do you handle state management in React applications?

Candidate: I usually use Redux for state management. It's what I'm most familiar with. Sometimes I use useState for simpler components. I know there are other options like MobX and Context, but I haven't used them much.

Interviewer: Could you explain your approach to testing?

Candidate: I write tests using Jest. I try to test the main functionality of components. Sometimes I struggle with writing good test cases, especially for complex components. I aim for around 60% coverage, though sometimes we don't reach that.

Interviewer: What's your experience with responsive design and accessibility?

Candidate: I use Bootstrap for most responsive design needs. It handles most cases well. For accessibility, I try to add alt tags to images and use semantic HTML when I remember to. I haven't done much testing with screen readers.

Interviewer: How do you stay updated with frontend technologies?

Candidate: I follow some developers on Twitter and occasionally read Medium articles. When my team decides to use a new technology, I learn it then. I've been meaning to try some newer frameworks but haven't had the time.

Interviewer: How do you debug frontend issues?

Candidate: I mainly use console.log for debugging. Sometimes I use the Chrome DevTools to inspect elements and check network requests. If I can't figure something out, I usually ask my team for help or check Stack Overflow.

Interviewer: What's your experience with build tools and bundlers?

Candidate: I've used Create React App for most of my projects. It handles all the webpack configuration for me. I know there are other tools like Vite, but I haven't tried them yet.

Interviewer: Do you have any questions for us?

Candidate: What's the tech stack you use here? And how big is the team?
"""
}

BAD_INTERVIEW = {
    "duration_minutes": 30,
    "position": "Frontend Engineer",
    "transcript": """
Interviewer: Hello! Could you introduce yourself and tell us about your background?

Candidate: Um, yeah, hi. I'm Chris. I've been doing some web development stuff for about 6 months, mostly following tutorials and building small projects.

Interviewer: Could you tell me about your experience with React?

Candidate: Well, I've watched some YouTube videos about it. I tried making a todo app but got stuck with the state management part. I mostly just copy and paste code from Stack Overflow when I need to do something.

Interviewer: How would you handle state management in a complex application?

Candidate: Uh, I guess I'd probably use variables? I'm not really sure about state management. I've heard of Redux but it seems really complicated. Usually I just put everything in one big component.

Interviewer: Could you explain the difference between controlled and uncontrolled components?

Candidate: Um... I'm not really sure what those are. Are they like different types of React components or something? Sorry, I should probably know this.

Interviewer: How do you approach testing in your projects?

Candidate: I don't really do testing. It seems like it takes too much time, and if the code works, it works, right? I usually just click around the page to see if it's broken.

Interviewer: What's your understanding of responsive design?

Candidate: That's when you make the website work on phones, right? I usually just make everything width: 100% and hope it works. CSS is pretty confusing to me.

Interviewer: How do you debug issues in your code?

Candidate: I put console.log everywhere until something works. If that doesn't help, I usually just delete the code and start over. Sometimes I'll post on Reddit asking for help.

Interviewer: What's your experience with version control?

Candidate: I've used GitHub desktop to upload code, but I always get confused when there are conflicts. I usually just create a new repository when that happens. Git commands are really confusing.

Interviewer: How do you stay updated with frontend technologies?

Candidate: I don't really keep up with it. There's too many new things coming out all the time. I just try to make things work however I can.

Interviewer: Do you have any questions for us?

Candidate: Not really. How long until I hear back about the job?
"""
} 