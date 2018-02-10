import matplotlib.pyplot as plt

class History():
    def __init__(self):
        self.history = {'val_loss': [],
                        'val_acc': [],
                        'tr_loss': [],
                        'tr_acc': []
                        }

    def __call__(self, epoch, val_loss, val_acc, tr_loss, tr_acc):
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['tr_loss'].append(tr_loss)
        self.history['tr_acc'].append(tr_acc)

        print('epoch: {:4d} \
               tr_loss: {:.5f} \
               tr_acc: {:.5f} \
               val_loss: {:.5f} \
               val_acc: {:.5f}'.format(epoch, 
                                       tr_loss, tr_acc,
                                       val_loss, val_acc))

    def plot(self):
        '''
        学習の進み具合を可視化
        '''
        v_loss = self.history['val_loss']
        t_loss = self.history['tr_loss']
        v_acc = self.history['val_acc']
        t_acc = self.history['tr_acc']
        plt.rc('font', family='serif')
        fig = plt.figure()
        plt.plot(range(len(v_loss)), v_loss,
                 label='validation_loss', color='red')
        plt.plot(range(len(t_loss)), t_loss,
                 label='training_loss', color='black')
        plt.xlabel('epochs')
        plt.ylabel('cross entropy')

        fig = plt.figure()
        plt.plot(range(len(v_acc)), v_acc,
                 label='validation_acc', color='red')
        plt.plot(range(len(t_acc)), t_acc,
                 label='training_acc', color='black')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.show()


class Real_time_plot():
    def __init__(self):
        fig = plt.figure()
        self.ax = fig.add_subplot(111, frameon=False)
        plt.ion()
        plt.grid()
        plt.show()

    def plot(self, data, pause_time = 60.0):
        plt.cla()
        self.ax.plot(data)
        plt.grid()
        plt.draw()
        plt.pause(1.0 / pause_time)

class EarlyStopping():
    def __init__(self, patience=0, verbose=0):
        self._step = 0
        self._loss = float('inf')
        self.patience = patience
        self.verbose = verbose

    def validate(self, loss):
        if self._loss < loss:
            self._step += 1
            if self._step > self.patience:
                if self.verbose:
                    print('early stopping')
                return True
        else:
            self._step = 0
            self._loss = loss

        return False
