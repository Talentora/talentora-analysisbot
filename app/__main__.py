from app import create_app

'''
Current implementation: After interview concludes, the interviee will be taken to a thank you page. 
This page URL will be the trigger urls to activate the AI model processing functions
'''

if __name__ == "__main__":

    app = create_app()
    app.run(
        debug=False,
    )

    
#fly.io https://fly.io/docs/launch/deploy/

"""
Deployment Backend
https://railway.app/
"""