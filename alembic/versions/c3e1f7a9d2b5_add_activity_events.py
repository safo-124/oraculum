"""add activity_events (admin Activity page: visitors and active users)

Revision ID: c3e1f7a9d2b5
Revises: b7f3d9a2c4e8
Create Date: 2026-11-18 12:00:00.000000

One row per page visit or chat question. No IP addresses and no question
text: visitors are identified by a salted hash of the client IP.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c3e1f7a9d2b5'
down_revision: Union[str, Sequence[str], None] = 'b7f3d9a2c4e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'activity_events',
        sa.Column('id', sa.Uuid(), primary_key=True),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('actor_kind', sa.String(length=16), nullable=False),
        sa.Column('visitor', sa.String(length=32), nullable=False),
        sa.Column('user_id', sa.Uuid(), nullable=True),
        sa.Column('path', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
    )
    op.create_index('ix_activity_events_kind', 'activity_events', ['kind'])
    op.create_index('ix_activity_events_actor_kind', 'activity_events', ['actor_kind'])
    op.create_index('ix_activity_events_visitor', 'activity_events', ['visitor'])
    op.create_index('ix_activity_events_user_id', 'activity_events', ['user_id'])
    op.create_index('ix_activity_events_created_at', 'activity_events', ['created_at'])


def downgrade() -> None:
    op.drop_table('activity_events')
