class VAE(nn.Module):
    '''
    An autoencoder, which will serve as part of the Generator in the GAN
    idims: input image dimensions [height, width]
    n_channels: number of color chanels
    n_filters: number of filter in layer directly adjacent to input/output
    z_dims: size of encoded representation
    
    
    Note: this assumes images are divisible by two with no remainder at least n_layers times
    '''
    def __init__(self, idims, n_channels, n_filters, z_dims):
        super().__init__()
        
        #ENCODE--------------------------------------------------------------------------------------------------------
        self.e_initial = ConvBlock(n_channels, n_filters, ks=4, stride=2, bn=False) #input conv block
        cdims = np.array(idims)/2
        cn_filters = n_filters
        
        e_layers = []
        while(max(cdims)>4):
            e_layers.append(ConvBlock(cn_filters, cn_filters*2, ks=4, stride=2))
            cn_filters *= 2
            cdims /=2
            
        self.encoding = nn.Sequential(*e_layers)
        self.e_means = ConvBlock(cn_filters, z_dims, ks=[int(x) for x in cdims], stride=1, pad=0, bn=False)
        self.e_stds = ConvBlock(cn_filters, z_dims, ks=[int(x) for x in cdims], stride=1, pad=0, bn=False)
               
        #DECODE--------------------------------------------------------------------------------------------------------
        self.d_initial = DeconvBlock(z_dims, cn_filters, ks=[int(x) for x in cdims], stride=1, pad=0)
        cdims *= 2
        
        d_layers = []
        while(max(cdims)<max(idims)):
            d_layers.append(DeconvBlock(cn_filters, cn_filters//2, ks=4, stride=2, pad=1))
            cn_filters //= 2
            cdims *= 2
            
        d_layers.append(DeconvBlock(n_filters, n_channels, ks=4, stride=2, pad=1))
            
        self.decoding = nn.Sequential(*d_layers)

        
    def forward(self, x):
        #Encoding
        x = self.e_initial(x)
        enc = self.encoding(x)
        #draw samples from distributions:
        mu = self.e_means(enc)
        var = self.e_stds(enc)
        samples = torch.normal(mu, var)
        
        #Decoding
        x = self.d_initial(samples)
        x = self.decoding(x)
    
        #TODO: Add decoder
        return mu, var, x