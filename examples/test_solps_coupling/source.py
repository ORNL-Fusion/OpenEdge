import random
random.seed(42)

# Base particles
base = [
(3.39700, -3.54075, -5.39584, 25.0707, -2.5234),
(3.39600, -3.54075, -3.39584, 20.0707, -2.5234),
(3.39500, -3.54075, -3.39584, 15.0707, -2.5234),
(3.39400, -3.54075, -3.39584, 30.0707, -2.5234),
]

particles = []
pid = 100000001

for i in range(50):
    bx, by, bvx, bvy, bvz = base[i % 4]
    # Small perturbations to position (within ~2mm) and velocity (~10%)
    x = bx + random.uniform(-0.002, 0.002)
    y = by + random.uniform(-0.002, 0.002)
    vx = bvx * random.uniform(0.9, 1.1)
    vy = bvy * random.uniform(0.9, 1.1)
    vz = bvz * random.uniform(0.9, 1.1)
    particles.append((pid + i, x, y, vx, vy, vz))

print('ITEM: TIMESTEP')
print('1')
print('ITEM: NUMBER OF ATOMS')
print(len(particles))
print('ITEM: BOX BOUNDS rr rr pp')
print('2 6')
print('-4 3.8')
print('-0.05 0.05')
print('ITEM: ATOMS id type x y z vx vy vz')
for pid, x, y, vx, vy, vz in particles:
    print(f'{pid} 1 {x:.5f} {y:.5f} 0 {vx:.5f} {vy:.4f} {vz:.4f}')
    " > /Users/42d/OpenEdge/examples/test_solps_coupling/input/particle.source"
