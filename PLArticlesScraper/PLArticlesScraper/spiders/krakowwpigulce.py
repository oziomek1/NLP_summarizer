# -*- coding: utf-8 -*-
import scrapy


class KrakowWPigulceSpider(scrapy.Spider):
    name = 'krakowwpigulce'
    allowed_domains = ['krakowwpigulce.pl']
    start_urls = ['https://krakowwpigulce.pl']

    custom_settings = {
        'DEPTH_LIMIT': 0
    }

    ARTICLES_URLS = "//article/a/@href"
    NEXT_PAGE_URL = "//div[contains(@class, 'pagination')]/ul/li/a/@href"

    TITLE_NODE = "//div[contains(@class, 'single_post')]//h1/text()"
    LEAD_NODE = "//div[contains(@class, 'single_post')]/div/p/strong/text()"
    TEXT_NODE = "//div[contains(@class, 'single_post')]/div/p/text()"

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
