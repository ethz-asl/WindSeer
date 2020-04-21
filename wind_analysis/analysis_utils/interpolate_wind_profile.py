import torch


def interpolate_wind_profile(x, y, x_new, device):
    """
    This function returns interpolated values of a set of 1-D functions at the desired query points x_new.
    """
    y_new = torch.zeros_like(torch.from_numpy(x_new))
    for i in range(x_new.shape[0]):
        pos = 0
        # find position of x_new within x elements
        for j in range(len(x)-1):
            if x[j] <= x_new[i] <= x[j+1]:
                pos = j
                break
            else:
                pos = j+1
        if pos >= len(x)-1:
            y_new[i] = y[-1]
        else:
            y_new[i] = y[pos] + (x_new[i]-x[pos])/(x[pos+1]-x[pos]) * (y[pos+1]-y[pos])
    return y_new.float().to(device)
