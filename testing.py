from db.session import get_session; 
with get_session() as s:
    print('Connected OK')