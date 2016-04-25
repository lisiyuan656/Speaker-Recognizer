import numpy

def vad_filter(frames):
    """Filter the frames by removing frames which has energy <= 0.01*the average energy of all the frames"""
    filter_coe = 0.01
    frame_energies = numpy.sum(numpy.square(frames), axis=1)
    energy_mean = numpy.mean(frame_energies)
    return numpy.delete(frames, [x for x in range(len(frame_energies)) if (frame_energies[x]<=energy_mean*filter_coe)], axis=0)
