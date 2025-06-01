import matplotlib.pyplot as plt
import pandas as pd

def visualize_resume_analysis(results):
    """Visualize resume analysis results."""
    # Prepare data storage
    data = {
        'Candidate': [],
        'Overall Match': [],
        'Skills Match': [],
        'Experience Match': [],
        'Education Match': []
    }

    # Populate data from results
    for candidate_name, result in results:
        data['Candidate'].append(candidate_name)
        data['Overall Match'].append(result['overall_match'])
        data['Skills Match'].append(result['skills_match'])
        data['Experience Match'].append(result['experience_match'])
        data['Education Match'].append(result['education_match'])

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(data)

    # Set the index to Candidate names
    df.set_index('Candidate', inplace=True)

    # Plotting
    ax = df.plot(kind='bar', figsize=(10, 6))
    plt.title('Resume Analysis Comparison')
    plt.ylabel('Match Percentage')
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.grid(axis='y')

    # Adding a legend
    plt.legend(title='Match Type')
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Assuming results are passed directly from the analysis
    from app.test.test_resumes.test_resume_analysis import create_sample_resumes
    results = create_sample_resumes()  # Get results from the analysis
    visualize_resume_analysis(results)  # Visualize the results 