#!/bin/bash
for filepath in updated_agents/*.zip; do
  python3 render_agent.py $filepath -r
done
