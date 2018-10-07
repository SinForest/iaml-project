import sys
import pickle
import matplotlib.pyplot as plt

fname = sys.argv[1]
title = sys.argv[2]

d = pickle.load(open(fname, 'rb'))

points = list(d.items())
points.sort(key=lambda x:x[0])
ep, v = zip(*points)
print(v)

val_prec = [x['val']['prec'] for x in v if "val" in x]
val_loss = [x['val']['loss'] for x in v if "val" in x]
val_schd = [x['val']['sched'] for x in v if "val" in x]
schd_xy  = [(x, y) for x, b, y in zip(ep, val_schd, val_loss) if b]

train_prec = []
train_loss = []
train_x    = []

ep = ep[:len(val_prec)]

fig, ax1 = plt.subplots()
plt.legend()
plt.xlabel("epochs")
plt.title(title)
ax2 = ax1.twinx()
for i, v in points:
    print(i, len(v))
    for n, dc in v.items():
        train_prec.append(dc['prec'])
        train_loss.append(dc['loss'])
    train_x.extend(list(map(lambda x: (i-1)+x/(len(v)+1), range(1, len(v)+1))))

ax1.plot(ep, val_prec, "-", color="red", label="validation accuracy", linewidth=3)
ax1.plot(train_x, train_prec, "-", color="magenta", label="training accuracy")
ax1.set_ylabel("accuracy")
ax2.plot(ep, val_loss, "-", color="blue", label="validation loss", linewidth=3)
ax2.plot(train_x, train_loss, "-", color="cyan", label="training loss") 
if schd_xy: 
    ax2.scatter(*zip(*schd_xy), marker="*", s=100, color="yellow", zorder=123)
ax2.set_ylabel("loss")

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

first_legend = plt.legend(handles1, labels1, loc=2, bbox_to_anchor=(0.0, 1.125), fancybox=True, framealpha=0.8)
ax2.add_artist(first_legend)

# Add the second legend as usual
ax2.legend(handles2, labels2,loc=0, bbox_to_anchor=(1.0, 1.125), fancybox=True, framealpha=0.8)
plt.tight_layout()
plt.savefig(fname + ".png")
plt.show()
"""
ax1.legend(loc=2, bbox_to_anchor=(0.0, 1.125), fancybox=True, framealpha=1)
ax2.legend(loc=0, bbox_to_anchor=(1.0, 1.125), fancybox=True, framealpha=1)
plt.show()
"""