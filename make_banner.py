import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.axis('off')

plt.text(0.5, 0.7, 'NANOMACHINE', ha='center', fontsize=28, fontweight='bold')
plt.text(0.5, 0.5, 'Hybrid RL × Swarm Intelligence System', ha='center', fontsize=14)
plt.text(0.5, 0.3, 'Multi-Agent Tissue Repair Research Framework', ha='center', fontsize=12)

plt.savefig('visuals/github_banner.png', dpi=300, bbox_inches='tight')
print('Banner created')
