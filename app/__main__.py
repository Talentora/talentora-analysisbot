from app import create_app

'''
Current implementation: After interview concludes, the interviewee will be taken to a thank you page. 
This page URL will be the trigger URLs to activate the AI model processing functions.
'''

if __name__ == "__main__":
    app = create_app()
    app.run(
        host='0.0.0.0',  # Make the server accessible externally
        port=5000,       # Specify the port
        debug=True,
    )
 
    
#fly.io https://fly.io/docs/launch/deploy/

"""
Deployment Backend
https://railway.app/
"""