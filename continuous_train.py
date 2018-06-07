from single_train import main
import logging


logger = logging.getLogger(__name__)


if __name__ == '__main__':

    logger.info('Start Continuous')
    main(continuous=True)
    logger.info('Script successfully finished...')
