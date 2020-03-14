# -*- coding: utf-8 -*-
import scrapy


class StrefainwestorowSpider(scrapy.Spider):
    name = 'strefainwestorow'
    allowed_domains = ['strefainwestorow.pl']
    start_urls = ['http://strefainwestorow.pl/artykuly']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//div[contains(@class, 'region-content')]/section/div/div[contains(@class, 'view-content')]//div/a/@href"
    NEXT_PAGE_URL = "//div[contains(@class, 'region-content')]/section/div/div[contains(@class, 'text-center')]/ul/li[contains(@class, 'next')]/a/@href"

    TITLE_NODE = "//div[contains(@class, 'panels-flexible-region-node-panel-center-inside')]/div[contains(@class, 'pane-node-title')]//div/h1/text()"
    LEAD_NODE = "//div[contains(@class, 'panels-flexible-region-node-panel-center-inside')]/div[contains(@class, 'pane-node-content')]//div/p/strong/text()"
    TEXT_NODE = "//div[contains(@class, 'panels-flexible-region-node-panel-center-inside')]/div[contains(@class, 'pane-node-content')]//div//text()"

    def parse(self, response):
        # iterate through links to articles
        for quote in response.xpath(self.ARTICLES_URLS):
            yield response.follow(quote, callback=self.parse_article)
            
        #iterate through next page links
        for next_page in response.xpath(self.NEXT_PAGE_URL):
            yield response.follow(next_page, self.parse)
            
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.xpath(self.TITLE_NODE).extract(),
            'lead': response.xpath(self.LEAD_NODE).extract(),
            'text': response.xpath(self.TEXT_NODE).extract(),
        }
