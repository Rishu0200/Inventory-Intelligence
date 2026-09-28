"""
scripts/create_user.py — create a user account (admin CLI tool, not a public endpoint).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import getpass
from db.session import get_session, init_db
from db.models import User
from auth.security import hash_password


def main():
    init_db()
    email = input("Email: ").strip()
    full_name = input("Full name: ").strip()
    role = input("Role [viewer/admin] (default viewer): ").strip() or "viewer"
    password = getpass.getpass("Password: ")
    confirm = getpass.getpass("Confirm password: ")

    if password != confirm:
        print("✗ Passwords don't match.")
        return

    with get_session() as s:
        if s.query(User).filter_by(email=email).first():
            print(f"✗ User with email {email} already exists.")
            return
        s.add(User(email=email, full_name=full_name, role=role,
                    hashed_password=hash_password(password), is_active=True))

    print(f"✓ Created user: {email} (role={role})")


if __name__ == "__main__":
    main()