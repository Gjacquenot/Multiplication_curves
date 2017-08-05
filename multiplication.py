#!/usr/bin/env python3
#
# This script creates multiplication curves, that exhibits
# mathmetical properties of multiplication
#
# Requires a Python interpreter with numpy and matplotlib
#
# Requires imagemagick to access convert when creating gif
# Requires ffmpeg to create mp4 video


import numpy as np


def create_multiplication_curve_coords(**kwargs):
    '''
    Creates the set of lines for a given `factor` and a given `base`
    '''
    factor = kwargs.get('factor', 2)
    n = kwargs.get('base', 100)
    x = np.arange(n)
    y = (factor * x) % n
    theta1 = 2 * np.pi * x / n
    theta2 = 2 * np.pi * y / n
    res = {'theta1': theta1, 'theta2': theta2,
           'x1': np.cos(theta1), 'x2': np.cos(theta2),
           'y1': np.sin(theta1), 'y2': np.sin(theta2)}
    return res


def get_filename(base=100, factor=2):
    '''
    Generates an output filename from `factor` and a given `base`
    '''
    return 'Mult_{0:05d}_{1:07d}.png'.format(base, int(1000 * factor))


def create_a_multiplication_figure(**kwargs):
    '''
    Generates an output filename from `factor`, a given `base` and
    '''
    import matplotlib.pyplot as plt
    factor = kwargs.get('factor', 2)
    base = kwargs.get('base', 100)
    filename = kwargs.get('filename', None)
    grid = kwargs.get('grid', False)
    text = kwargs.get('text', False)
    dpi = kwargs.get('dpi', 600)
    if filename == '':
        filename = get_filename(base=base, factor=factor)
    fig = plt.figure()
    fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    theta = np.linspace(0, 2 * np.pi, base + 1)
    x = np.cos(theta)
    y = np.sin(theta)
    ax.plot(x, y, linewidth=0.5, color='k')
    res = create_multiplication_curve_coords(base=base, factor=factor)
    X1, X2, Y1, Y2 = res['x1'], res['x2'], res['y1'], res['y2']
    for x1, x2, y1, y2 in zip(X1, X2, Y1, Y2):
        ax.plot([x1, x2], [y1, y2], linewidth=0.25, color='k')
    if text:
        plt.text(0.02, 0.92, 'Base = {0}\nFactor = {1}'.format(base, np.round(1e6 * factor)/1e6),
            fontsize=2.5,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes)
        def add_numbers(vec):
            for i in vec:
                plt.text(1.07 * np.cos(2.0*np.pi*i/base),
                         1.07 * np.sin(2.0*np.pi*i/base),
                         str(i),
                         fontsize=2,
                         horizontalalignment='center',
                         verticalalignment='center')
        def multiply(n):
            total = 1
            for i in n:
                total *= i
            return total
        if base < 36:
            add_numbers(range(base))
        else:
            pfactors = prime_factors(base)
            print(pfactors)
            nfactors = len(pfactors)
            if nfactors == 1:
                add_numbers(range(pfactors[0]))
            elif nfactors == 2:
                add_numbers([pfactors[0] * e for e in range(pfactors[1])])
            elif nfactors == 3:
                add_numbers([pfactors[0] * pfactors[1] * e for e in range(pfactors[2])])
            elif nfactors == 4:
                add_numbers([pfactors[0] * pfactors[1] * pfactors[2] * e for e in range(pfactors[3])])
            else:
                add_numbers(multiply(pfactors[:-2]) * e for e in range(multiply(pfactors[-2:])))
    if grid:
        ax.grid(True)
    else:
        ax.axis('off')
    if filename:
        plt.savefig(filename, dpi=dpi)
        plt.close()
    else:
        plt.show()


def prime_factors(x):
    factorlist = []
    loop = 2
    while loop <= x:
        if x%loop == 0:
            x//=loop
            factorlist.append(loop)
        else:
            loop += 1
    return factorlist


def create_a_multiplication_figure_to_expand(kwargs):
    create_a_multiplication_figure(**kwargs)


def create_animated_gif(filename='multiplication_curve.gif', **kwargs):
    import subprocess
    pngs = kwargs.get('pngs', None)
    continuous = kwargs.get('continuous', False)
    if pngs is None:
        from glob import glob
        pngs = glob('*.png')
    if continuous:
        pngs += pngs[-2:0:-1]
    cmd = 'convert -antialias -density 100 -delay 120 '
    cmd += ' '.join(pngs)
    cmd += ' ' + filename
    subprocess.check_output(cmd.split(' '))


def create_animated_mp4(filename='multiplication_curve.mp4', **kwargs):
    import subprocess
    import os
    pngs = kwargs.get('pngs', None)
    framerate = kwargs.get('framerate', 12)
    continuous = kwargs.get('continuous', False)
    if pngs is None:
        from glob import glob
        pngs = glob('*.png')
    if continuous:
        pngs += pngs[-2:0:-1]
    infile = open('tmp.txt','w')
    for png in pngs:
        infile.write('file ' + png + '\n')
        infile.write('duration ' + str(1.0/framerate) + '\n')
    infile.write('file ' + pngs[-1] + '\n')
    infile.close()
    cmd = 'ffmpeg -f concat -i tmp.txt ' + filename
    subprocess.check_output(cmd.split(' '))
    os.remove('tmp.txt')


def get_description():
    from textwrap import dedent
    description = """
        Generate multiplication curves given a factor and a basis

        factor * [0, 1, 2, ..., base[ % base
        """
    return dedent(description)


def get_epilog():
    from textwrap import dedent
    epilog = """
        # Display help
        python multiplication.py -h

        # Create a single simple image with default parameters
        python multiplication.py -o test.png

        # Create a single simple image with factor 9 and base 360
        python multiplication.py -b 360 -f 9 -o test.png

        # Create a movie with 21 frames where factor goes from 2 to 3 on a 360 basis with text. Parallelism is used
        python multiplication.py -t -b 360 -f 2--3 -n 21 -p -o mult.mp4

        # Same as previous but with fully developped argument
        python multiplication.py --text --base 360 --factor 2--3 --number 21 --parallel --output mult.mp4
        """
    return dedent(epilog)


def main():
    import argparse
    import os
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    parser = argparse.ArgumentParser(description=get_description(),
                                     epilog=get_epilog(),
                                     formatter_class=CustomFormatter)
    pa = parser.add_argument
    pa('-b', '--base', type=int, help='positive integer representing the base to use. Reasonnable value is 100', default=100)
    pa('-f', '--factor', type=str, help='positive float factor to use. When creating an animation, start and end values are separated with -- (e.g. 2--3)' , default=2.0)
    pa('-o', '--output', type=str, default=None, help='name of the generated file. If not provided, result will display on screen. Extensions can be png, gif, mp4')
    pa('-t', '--text', action='store_true', help='boolean used to display base and factor on each generated image.')
    pa('-d', '--dpi', type=int, help='number of pixels of the generated output. Default is 600', default=600)
    pa('-n', '--number', type=int, help='number of pictures to generate between two complex numbers. Default is 2', default=2)
    pa('-p', '--parallel', action='store_true', help='boolean used to create images in a parallel way. It used the (n-1) cores. Default is False')
    pa('-c', '--continuous', action='store_true', help='boolean used to create continuous animation that can loop. Default is False')
    pa('-r', '--remove_pngs', action='store_true', help='boolean used to remove generated image with an animation or a movie. Default is False')
    args = parser.parse_args()
    if args.output is None:
        output = None
    else:
        output = args.output
    if output:
        output_lower = output.lower()
        if output_lower.endswith('.gif') or output_lower.endswith('.mp4'):
            f = args.factor.split('--')
            v = [e for e in np.linspace(float(f[0]), float(f[1]), args.number)]
            filenames = [get_filename(base=args.base, factor=k) for k in v]
            if args.parallel:
                from multiprocessing import cpu_count
                from multiprocessing import Pool
                ncores = max(1, cpu_count()-1)
                listOfInputs = [{'base':args.base, 'factor':f, 'text':args.text, 'filename':k, 'dpi':args.dpi} for f, k in zip(v, filenames)]
                p = Pool(ncores)
                p.map(create_a_multiplication_figure_to_expand, listOfInputs)
            else:
                for k, filename in zip(v, filenames):
                    create_a_multiplication_figure(base=args.base, factor=k, text=args.text, filename=filename, dpi=args.dpi)
            if output_lower.endswith('.gif'):
                create_animated_gif(filename=output, pngs=filenames, continuous=args.continuous)
            elif output_lower.endswith('.mp4'):
                create_animated_mp4(filename=output, pngs=filenames, continuous=args.continuous)
            if args.remove_pngs:
                for f in set(filenames):
                    os.remove(f)
        else:
            create_a_multiplication_figure(base=args.base, factor=float(args.factor), text=args.text, filename=output, dpi=args.dpi)
    else:
        create_a_multiplication_figure(base=args.base, factor=float(args.factor), text=args.text, filename=args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
