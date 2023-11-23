from tools.ops import *

def D_net(x_init, sn, ch, scope, reuse):
    channel = ch
    with tf.variable_scope(scope, reuse=reuse):
        x = conv(x_init, channel, kernel=7, stride=1, sn=sn, scope='conv_0')
        x = lrelu(x)
        for i in range(3):
            x = conv(x, channel, kernel=3, stride=2,  sn=sn, scope='conv_s2_' + str(i))
            x = LADE_D(x, sn, scope+'a'+str(i))
            x = lrelu(x)

            x = conv(x, channel * 2, kernel=3, stride=1,  sn=sn, scope='conv_s1_' + str(i))
            x = LADE_D(x, sn, scope+'b'+str(i))
            x = lrelu(x)
            channel = channel * 2
        x = conv(x, channels=1, kernel=1, stride=1, sn=sn, scope='D_logit')
        return x







