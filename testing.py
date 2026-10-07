from agents.demand_agent import demand_agent_node 
from agents.supplier_agent import supplier_agent_node
print(supplier_agent_node({'sku_id': 'RSH-001', 'query': 'best supplier'})['tool_result'][:200])
print(demand_agent_node({'sku_id': '', 'query': 'forecast demand'})['tool_result'][:200])